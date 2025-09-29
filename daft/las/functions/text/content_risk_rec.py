# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import tracking_usage
from daft.las.infra.content_security import ContentSecurityConfig, get_content_security_service

logger = logging.getLogger(__name__)

SERVICE = "text_risk"


class RetryableError(Exception):
    pass


class ContentRiskRec(Operator):
    """**文本内容风险识别处理器，支持多维度安全检测**

    **核心功能：**
    - 多风险类型检测
    - 三级风险判定：
        - BLOCK: 不安全样本
        - PASS: 安全样本
        - REVIEW: 建议人工审核
    - 支持自定义业务场景配置
    - 提供详细检测结果输出

    **性能说明：**
    - 支持批量处理，建议根据服务QPS配置并发量

    Notes
    -----
    算子使用前置条件：开通业务风险识别产品-文本风险识别，产品链接见：https://www.volcengine.com/product/business-security
    """  # noqa: D415, D416

    def __init__(
        self,
        app_id: int,
        biztype: str,
        timeout: int = 120,
        **kwargs: Any,
    ):
        """初始化文本内容风险识别处理器

        Args:
            app_id: 在业务风险识别产品中开通的应用ID
            biztype: 在业务风险识别产品中配置的场景
            timeout: 超时时间设置，单位（s）
                设置从发送请求到接收到结果的超时设置，如果是离线任务，为了防止数据处理阻塞，可以适当的增加该值
        """  # noqa: D415
        super().__init__(**kwargs)
        self.app_id = app_id
        self.biztype = biztype
        self.timeout = timeout
        self._content_service = get_content_security_service(ContentSecurityConfig.from_env())
        self.executor = ThreadPoolExecutor(max_workers=5)

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="Content Security")

    @retry(  # type: ignore[misc]
        wait=wait_exponential(multiplier=1.5, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(RetryableError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def async_text_risk(self, params: dict[str, Any]) -> dict[str, str]:
        """Asynchronously call the text risk detection service with retry and timeout."""
        req = {
            "AppId": self.app_id,
            "Service": SERVICE,
            "Parameters": json.dumps(params),
        }
        try:
            loop = asyncio.get_running_loop()
            res = await asyncio.wait_for(
                loop.run_in_executor(self.executor, partial(self._content_service.text_slice_risk, {}, req)),
                timeout=self.timeout,
            )
            if isinstance(res, Exception):
                raise RetryableError(str(res))
            result = res.get("Result", {})
            message = result.get("Message", "unknown")
            data = result.get("Data", {}) if message == "success" else {}
            return {
                "FinalLabel": data.get("FinalLabel", ""),
                "Decision": data.get("Decision", ""),
                "Message": message,
                "risk_result": json.dumps(res),
            }
        except Exception as e:
            logger.warning("Request failed: %s", e)
            self._refresh_service()
            raise RetryableError(str(e))

    async def concurrent_requests(self, batch: pa.RecordBatch) -> list[dict[str, Any]]:
        """Execute concurrent requests with processed results."""
        if batch.num_rows == 0:
            return []

        records = batch.to_pylist()
        parameters_list = []
        for record in records:
            params = {k: v for k, v in record.items() if v is not None}
            params.setdefault("operate_time", int(time.time()))
            params.setdefault("account_id", "user")
            params["biztype"] = self.biztype
            parameters_list.append(params)

        results = await asyncio.gather(
            *[self.async_text_risk(params) for params in parameters_list], return_exceptions=True
        )
        final_results: list[dict[str, str]] = []
        for res in results:
            if isinstance(res, Exception):
                final_results.append(
                    {
                        "FinalLabel": "",
                        "Decision": "",
                        "Message": str(res),
                        "risk_result": "{}",
                    }
                )
            else:
                final_results.append(res)  # type: ignore
        return final_results

    def _refresh_service(self) -> None:
        """Refresh the content security service client."""
        self._content_service = get_content_security_service(ContentSecurityConfig.from_env())

    def __del__(self) -> None:
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)

    def transform(
        self,
        text_col: pa.Array,
        account_id_col: pa.Array | None = None,
        operate_time_col: pa.Array | None = None,
        nick_name_col: pa.Array | None = None,
        signature_col: pa.Array | None = None,
        text_type_col: pa.Array | None = None,
        session_id_col: pa.Array | None = None,
    ) -> pa.Array:
        """对输入文本进行内容风险识别

        Args:
            text_col: 输入文本列
            account_id_col: (可选) 文本发送者的ID列
            operate_time_col: (可选) 用户发送文本的秒级时间戳列
            nick_name_col: (可选) 文本发送者的用户昵称列
            signature_col: (可选) 用户的个性签名列
            text_type_col: (可选) 文本内容的类型列 (例如 "prompt", "response")
            session_id_col: (可选) AIGC对话场景下的对话轮次ID列

        Returns:
            一个结构体数组，包含风险识别结果
                - FinalLabel: 最终风险标签
                - Decision: 判定结果 (PASS, BLOCK, REVIEW)
                - Message: 服务端返回信息
                - risk_result: 完整的JSON格式检测结果
        """  # noqa: D415
        if len(text_col) == 0:
            return pa.array([], type=self.__return_column_type__())

        arrays = [text_col]
        names = ["text"]

        def add_col_if_not_none(col_arr: pa.Array | None, name: str) -> None:
            if col_arr is not None:
                arrays.append(col_arr)
                names.append(name)

        add_col_if_not_none(account_id_col, "account_id")
        add_col_if_not_none(operate_time_col, "operate_time")
        add_col_if_not_none(nick_name_col, "nickname")
        add_col_if_not_none(signature_col, "signature")
        add_col_if_not_none(text_type_col, "text_type")
        add_col_if_not_none(session_id_col, "session_id")

        batch = pa.RecordBatch.from_arrays(arrays, names=names)
        all_results = asyncio.run(self.concurrent_requests(batch))

        return pa.array(all_results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.struct(
            [
                pa.field("FinalLabel", pa.string()),
                pa.field("Decision", pa.string()),
                pa.field("Message", pa.string()),
                pa.field("risk_result", pa.string()),
            ]
        )
