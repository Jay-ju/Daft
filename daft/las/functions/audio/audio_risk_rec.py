# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import tracking_usage
from daft.las.infra.content_security import ContentSecurityConfig, get_content_security_service

logger = logging.getLogger(__name__)

SERVICE = "audio_risk"


class RetryableError(Exception):
    pass


class AudioRiskRec(Operator):
    """**音频内容风险识别处理器，支持多格式音频安全检测**

    **核心功能：**
    - 支持多种音频格式（mp3, m4a, wav, wma, amr, aac, ogg）
    - 支持大文件（时长≤5小时，≤512MB）
    - 三级风险判定：
        - BLOCK: 不安全样本
        - PASS: 安全样本
        - REVIEW: 建议人工审核
    - 支持自定义业务场景配置
    - 提供详细检测结果输出

    **性能说明：**
    - 支持批量处理，建议根据服务QPS配置并发量
    - 异步接口，自动轮询获取检测结果

    Notes
    -----
    算子使用前置条件：开通业务风险识别产品-音频风险识别-音频点播服务，产品链接见：https://www.volcengine.com/product/business-security
    """  # noqa: D415, D416

    def __init__(
        self,
        app_id: int,
        biztype: str,
        result_type: int = 0,
        timeout: int = 120,
        poll_interval: int = 10,
        num_coroutines: int = 5,
        **kwargs: Any,
    ):
        """初始化音频内容风险识别处理器

        Args:
            app_id: 在业务风险识别产品中开通的应用ID
            biztype: 在业务风险识别产品中配置的音频风险识别场景
            result_type: 切片内容类型
                0 表示仅返回违规的音频切片
                1 表示返回所有切片检测结果
                可选值: [0, 1]
            timeout: 超时时间设置，单位（s）
                设置从发送请求到接收到结果的超时设置，如果是离线任务，为了防止数据处理阻塞，可以适当的增加该值
            poll_interval: 轮询间隔设置，单位（s）
                调用主动查询接口时，设置的轮询间隔，请根据主动查询接口QPS以及性能需求来设置
        """  # noqa: D415
        super().__init__(**kwargs)

        self.app_id = app_id
        self.biztype = biztype
        self.result_type = result_type
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._content_service = get_content_security_service(ContentSecurityConfig.from_env())
        self.async_result_timeout = self.timeout - 1 if self.timeout > 1 else self.timeout
        self.num_coroutines = num_coroutines

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="Content Security")

    @retry(  # type: ignore[misc]
        wait=wait_exponential(multiplier=1.5, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(RetryableError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def async_audio_risk(self, params: dict[str, Any]) -> dict[str, str]:
        """Asynchronously call the audio risk detection service and poll for results."""
        req = {
            "AppId": self.app_id,
            "Service": SERVICE,
            "Parameters": json.dumps(params),
        }
        req_get_result = {
            "AppId": self.app_id,
            "Service": SERVICE,
            "DataId": params["data_id"],
        }
        try:
            # Submit task
            _resp = self._content_service.async_audio_risk({}, req)
            logger.info("async_audio_risk response is %s", _resp)
            # Poll for result
            start_time = time.time()
            while time.time() - start_time < self.async_result_timeout:
                result = self._content_service.audio_result(req_get_result, {})
                logger.info("audio_result req is %s, response is %s", req_get_result, result)
                if result.get("Result", {}).get("Data"):
                    return self._parse_result(result)
                await asyncio.sleep(self.poll_interval)
            raise TimeoutError("Async result polling timeout")
        except Exception as e:
            logger.warning("Request failed: %s", e)
            self._refresh_service()
            raise RetryableError(e)

    async def concurrent_requests(self, batch: pa.RecordBatch) -> list[dict[str, Any]]:
        """Batch concurrent requests to the audio risk detection service."""
        if batch.num_rows == 0:
            return []

        records = batch.to_pylist()
        parameters_list = []
        for record in records:
            params = {k: v for k, v in record.items() if v is not None}
            params.setdefault("operate_time", int(time.time()))
            params.setdefault("account_id", "user")
            params["result_type"] = self.result_type
            params["biztype"] = self.biztype
            parameters_list.append(params)

        semaphore = asyncio.Semaphore(self.num_coroutines)

        async def bounded_async_audio_risk(params: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await self.async_audio_risk(params)

        results = await asyncio.gather(
            *[bounded_async_audio_risk(params) for params in parameters_list], return_exceptions=True
        )
        final_results: list[dict[str, str]] = []
        for res in results:
            if isinstance(res, Exception):
                final_results.append(
                    {
                        "Decision": "",
                        "Message": str(res),
                        "risk_result": "{}",
                    }
                )
            else:
                final_results.append(res)  # type: ignore
        return final_results

    def _parse_result(self, res: dict[str, Any]) -> dict[str, str]:
        """Parse the service response into structured fields."""
        try:
            result = res.get("Result", {})
            message = result.get("Message", "unknown")
            data = result.get("Data", {}) if message == "success" else {}
            return {
                "Decision": data.get("Decision", ""),
                "Message": message,
                "risk_result": json.dumps(res, ensure_ascii=False),
            }
        except Exception as e:
            logger.info("Result processing error: %s", e)
            return {"Decision": "", "Message": str(e), "risk_result": json.dumps(res, ensure_ascii=False)}

    def _refresh_service(self) -> None:
        """Refresh the content security service client."""
        self._content_service = get_content_security_service(ContentSecurityConfig.from_env())

    def transform(
        self,
        audio_id_col: pa.Array,
        audio_url_col: pa.Array,
        audio_title_col: pa.Array | None = None,
        account_id_col: pa.Array | None = None,
        operate_time_col: pa.Array | None = None,
    ) -> pa.Array:
        """对输入音频进行内容风险识别

        Args:
            audio_id_col: 输入音频唯一标志列
            audio_url_col: 音频内容所在的列，支持可直接访问的 url 地址
            audio_title_col: (可选) 音频标题列
            account_id_col: (可选) 音频发送者的ID列
            operate_time_col: (可选) 用户发送音频的秒级时间戳列

        Returns:
            一个结构体，包含风险识别结果
                - Decision: 判定结果 (PASS, BLOCK, REVIEW)
                - Message: 服务端返回信息
                - risk_result: 完整的 JSON 格式检测结果
        """  # noqa: D415
        if len(audio_id_col) == 0:
            return pa.array([], type=self.__return_column_type__())

        arrays = [audio_id_col, audio_url_col]
        names = ["data_id", "url"]

        def add_col_if_not_none(col_arr: pa.Array | None, name: str) -> None:
            if col_arr is not None:
                arrays.append(col_arr)
                names.append(name)

        add_col_if_not_none(audio_title_col, "audio_title")
        add_col_if_not_none(account_id_col, "account_id")
        add_col_if_not_none(operate_time_col, "operate_time")

        batch = pa.RecordBatch.from_arrays(arrays, names=names)
        all_results = asyncio.run(self.concurrent_requests(batch))

        return pa.array(all_results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.struct(
            [
                pa.field("Decision", pa.string()),
                pa.field("Message", pa.string()),
                pa.field("risk_result", pa.string()),
            ]
        )
