# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import os
import random
import re
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pdfplumber
from aiolimiter import AsyncLimiter

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import tracking_usage
from daft.las.infra.visual_service import VisualServiceConfig, get_visual_service
from daft.las.io import mkdirs, upload_file

logger = logging.getLogger(__name__)


class PDFParse(Operator):
    """**PDF 智能文档解析处理器，支持多维度内容提取与结构化输出**

    **核心功能**
    - 高精度 PDF 文本提取
    - 输出 markdown 格式文本
    - 保留原始文档结构与格式
    - 支持表格、图片等内容识别
    - 提供详细解析结果与 TOS 存储选项

    **性能说明**
    - 服务 QPS 较低，建议设置并发量为1
    - 大文档处理可能需要较长时间

    Notes
    -----
    算子使用前置条件：开通视觉智能产品-文字识别-智能文档解析服务，产品链接见：https://www.volcengine.com/docs/86081/1804813
    """  # noqa: D415, D416

    def __init__(
        self,
        input_type: str,
        output_tos_path: str = "",
        version: str = "v3",
        file_type: str = "pdf",
        page_start: int = 0,
        page_parsed_num: int = -1,
        page_batch: int = 300,
        parse_mode: str = "auto",
        table_mode: str = "markdown",
        filter_header: str = "true",
        timeout: int = 120,
        qps: int = 2,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        """初始化 PDF 智能文档解析处理器。

        Args:
            input_type: 输入类型，支持 "url"（文件链接）或 "base64"（Base64 编码内容）。
                可选值：["url", "base64"]
            output_tos_path: 解析内容存储到的 TOS 目录。
                解析内容保存到的 TOS 目录下的 txt 文件夹中，
                同时会将PDF中的图片转存到 TOS目录下的 images 文件夹中，
                并在保存的 txt 文件中使用图片 TOS 路径。
                如果该值设置为空，则不需要将解析内容存储到 TOS 中。
                默认值：""
            version: 解析服务版本。
                可选值：["v3"]
                默认值："v3"
            file_type: 文件类型。
                可选值：["pdf"]
                默认值："pdf"
            page_start: 起始解析页码，从 0 开始。
                默认值：0
            page_parsed_num: 解析的页数，-1 表示解析全部。
                默认值：-1
            page_batch: 批量解析的页数。
                默认值：300
            parse_mode: 解析模式。
                可选值：["auto", "fast", "accurate"]
                默认值："auto"
            table_mode: 表格解析模式。
                可选值：["markdown", "html", "excel"]
                默认值："markdown"
            filter_header: 是否过滤页眉页脚。
                可选值：["true", "false"]
                默认值："true"
            timeout: 超时时间，单位为秒。
                默认值：120
            qps: 请求 QPS 限制。
                默认值：2
            max_retries: 最大重试次数。
                默认值：3
            **kwargs: 其他参数，透传给父类。
        """  # noqa: D415
        super().__init__(**kwargs)

        if input_type not in ("url", "base64"):
            raise ValueError("input_type must be either 'url' or 'base64'")

        self.input_type = input_type
        self.output_tos_path = output_tos_path.strip("/") if output_tos_path else ""
        self.version = version
        self.file_type = file_type
        self.page_start = page_start
        self.page_parsed_num = page_parsed_num
        self.page_batch = min(page_batch, 300)
        self.parse_mode = parse_mode
        self.table_mode = table_mode
        self.filter_header = filter_header
        self.timeout = timeout
        self.qps = qps
        self.max_retries = max_retries

        if self.output_tos_path:
            self.output_image_tos_path = self.output_tos_path + "/images"
            self.output_md_tos_path = self.output_tos_path + "/txt"
            self.output_detail_tos_path = self.output_tos_path + "/detail"
            mkdirs(self.output_image_tos_path)
            mkdirs(self.output_md_tos_path)
            mkdirs(self.output_detail_tos_path)

        self.base_delay = 1.5
        self._visual_service = get_visual_service(VisualServiceConfig.from_env())
        self.qps_limiter = AsyncLimiter(self.qps, 1)

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="visual service")

    def get_pdf_page_count_from_base64(self, base64_str: str) -> int:
        try:
            pdf_bytes = base64.b64decode(base64_str)
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                return len(pdf.pages)
        except (binascii.Error, pdfplumber.PDFSyntaxError):
            logger.exception("Get pdf page count from base64 error")
            return 0

    def get_pdf_page_count_from_url(self, url: str) -> int:
        try:
            response = httpx.get(url, timeout=10)
            response.raise_for_status()

            if "application/pdf" not in response.headers.get("Content-Type", ""):
                raise ValueError("URL did not return a PDF file")

            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                page_count = len(pdf.pages)
                logger.info("PDF page count: %d", page_count)
                return page_count

        except httpx.RequestError:
            logger.exception("Network request failed for url: %s", url)
        except Exception:
            logger.exception("Error processing PDF for url: %s", url)
        return 0

    def get_pdf_page_count(self, base64_str: str | None, url_str: str | None) -> int:
        if base64_str:
            return self.get_pdf_page_count_from_base64(base64_str)
        if url_str:
            return self.get_pdf_page_count_from_url(url_str)
        raise ValueError("base64_str or url_str must be provided.")

    async def async_ocr_pdf(self, detector: Any, req: dict[str, Any]) -> Any:
        for attempt in range(1, self.max_retries + 1):
            try:
                async with self.qps_limiter:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, partial(detector.ocr_pdf, req))

                if not isinstance(result, dict):
                    raise RuntimeError(f"OCR service returned non-dict response: {result}")

                code = result.get("code")
                msg = result.get("message", "")
                data = result.get("data") or {}
                markdown = data.get("markdown") if isinstance(data, dict) else None

                # Success conditions: explicit success code or message + markdown present
                if code == 10000 or (msg == "Success" and markdown not in (None, "")):
                    return result

                # Retryable conditions: known retryable codes or empty markdown indicating transient/QPS issues
                if self._is_retryable_error(code) or (markdown in (None, "") and code in (None, 50429, 50500, 50501)):
                    logger.warning(
                        "Transient OCR error (code=%s, msg=%s, attempt=%d/%d, req=%s)",
                        code,
                        msg,
                        attempt,
                        self.max_retries,
                        self._summarize_req(req),
                    )
                    if attempt == self.max_retries:
                        raise RuntimeError(f"OCR service error after retries: code={code}, message={msg}")
                    await asyncio.sleep(self.base_delay * (2**attempt))
                    self._refresh_service()
                    continue

                # Non-retryable: authentication/args/decode/size etc.
                logger.error("Non-retryable OCR error (code=%s, msg=%s, req=%s)", code, msg, self._summarize_req(req))
                raise RuntimeError(f"OCR service error code={code}, message={msg}")

            except (TimeoutError, ConnectionError) as e:
                logger.warning(
                    "Network/timeout in OCR (attempt %d/%d, req=%s): %s",
                    attempt,
                    self.max_retries,
                    self._summarize_req(req),
                    e,
                )
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(self.base_delay * (2**attempt))
                self._refresh_service()
            except Exception as e:
                # Handle rate limiting/slowdown hints
                emsg = str(e).lower()
                if "slowdown" in emsg or "rate limit" in emsg or "429" in emsg:
                    logger.warning(
                        "Rate limit/slowdown in OCR (attempt %d/%d, req=%s): %s",
                        attempt,
                        self.max_retries,
                        self._summarize_req(req),
                        e,
                    )
                    if attempt == self.max_retries:
                        raise
                    await asyncio.sleep(self.base_delay * (2**attempt))
                    self._refresh_service()
                    continue
                logger.exception("Unexpected error in PDF parse")
                raise
        return None

    async def _process_pdf_source(
        self, base64_str: str | None, url_str: str | None, file_name_str: str | None
    ) -> tuple[str, dict[str, Any]]:
        parameters_list = []
        if self.page_parsed_num == -1:
            try:
                page_number = self.get_pdf_page_count(base64_str, url_str)
                page_batch = self.page_batch
                logger.info("Get pdf page count: %d", page_number)
            except Exception:
                page_number = 300
                page_batch = 300
                logger.exception("Get pdf page count error. Set page_number:%d; page_batch:%d", page_number, page_batch)
        else:
            page_number = self.page_parsed_num
            page_batch = min(page_number, self.page_batch)
            logger.info("Set page_number:%d; page_batch:%d", page_number, page_batch)

        if page_number == 0:
            page_number = 300
            page_batch = 300
            logger.info("Get page number: 0. Set page_number:%d; page_batch:%d", page_number, page_batch)

        # Respect page_start and compute batches with correct remaining page_num
        effective_start = max(0, self.page_start)
        # When page_parsed_num == -1, parse_limit is total pages; otherwise parse pages starting from effective_start
        parse_limit = page_number if self.page_parsed_num == -1 else effective_start + page_number
        page_end = effective_start
        while page_end < parse_limit:
            page_begin = page_end
            remaining = parse_limit - page_begin
            current_batch = min(page_batch, remaining)
            page_end = page_end + current_batch
            params: dict[str, str] = {}
            if base64_str is not None:
                params["image_base64"] = str(base64_str)
            elif url_str is not None:
                params["image_url"] = str(url_str)
            params.update(
                {
                    "page_start": str(page_begin),
                    "page_num": str(current_batch),
                    "version": str(self.version),
                    "file_type": str(self.file_type),
                    "parse_mode": str(self.parse_mode),
                    "table_mode": str(self.table_mode),
                    "filter_header": str(self.filter_header),
                }
            )
            parameters_list.append(params)

        # Dispatch all page requests; AsyncLimiter enforces QPS across tasks
        tasks = [
            asyncio.wait_for(self.async_ocr_pdf(self._visual_service, params), timeout=self.timeout)
            for params in parameters_list
        ]
        results_total = await asyncio.gather(*tasks, return_exceptions=True)

        return await self._process_results(results_total, file_name_str)

    def _transform_md_2_plain_text(self, markdown: str) -> str:
        pattern = r"!\[fig_.*?\]\([^\)]*\)[\n| ]*"
        return re.sub(pattern, "", markdown)

    def _extract_image_filename(self, markdown: str) -> list[str]:
        pattern = re.compile(r"!\[fig_\d+]\(([^)]+)\)")
        return [match.group(1) for match in pattern.finditer(markdown)]

    async def _async_transform_md(self, markdown: str, output_parsed_file_name: str) -> tuple[str, dict[str, Any], str]:
        download_report: dict[str, Any] = {"total": 0, "success": 0, "failed": [], "duplicates": 0}

        def replacer(match: re.Match[str]) -> str:
            nonlocal download_report
            fig_desc = match.group(1)
            url = match.group(2)
            filename = os.path.basename(urlparse(url).path)
            try:
                response = httpx.get(url, timeout=(3.05, 30), headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()

                with tempfile.TemporaryDirectory(dir="/tmp") as temp_sub_dir:
                    tmp_file_name = Path(temp_sub_dir) / filename
                    with tmp_file_name.open("wb") as f:
                        f.write(response.content)
                    upload_file(str(tmp_file_name), f"{self.output_image_tos_path}/{tmp_file_name.name}")

                download_report["success"] += 1
            except Exception as e:
                download_report["failed"].append({"url": url, "error": str(e)})

            return f"![{fig_desc}]({filename})"

        pattern = re.compile(r"!\[(fig_[^]]*)]\((https?://[^)]+)\)")
        post_process_markdown = pattern.sub(replacer, markdown)

        with tempfile.TemporaryDirectory(dir="/tmp") as temp_sub_dir:
            tmp_md_file_name = Path(temp_sub_dir) / output_parsed_file_name
            with tmp_md_file_name.open("w") as f:
                f.write(post_process_markdown + "\n")
            upload_file(str(tmp_md_file_name), f"{self.output_md_tos_path}/{tmp_md_file_name.name}")

        parsed_file_tos_path = f"{self.output_md_tos_path.rstrip('/')}/{output_parsed_file_name}"
        logger.info("Download images report: %s", download_report)
        return post_process_markdown, download_report, parsed_file_tos_path

    async def _process_results(self, results: list[Any], file_name_str: str | None) -> tuple[str, dict[str, Any]]:
        markdown_list = []
        detail_list = []

        for res in results:
            if isinstance(res, Exception):
                markdown_list.append("")
                continue
            try:
                code = res.get("code")
                msg = res.get("message")
                if code == 10000 or msg == "Success":
                    data = res.get("data", {}) or {}
                    markdown_list.append(data.get("markdown", ""))
                    if data.get("detail"):
                        detail_list.extend(json.loads(data["detail"]))
            except Exception:
                logger.exception("Get detail & markdown from %s error", res)
                markdown_list.append("")

        markdown = "\n".join(markdown_list)
        markdown_plain_text = self._transform_md_2_plain_text(markdown)

        result: dict[str, Any] = {
            "parsed_origin_text": markdown,
            "parsed_plain_text": markdown_plain_text,
            "parsed_detail": detail_list,
            "parsed_file_path": "",
            "parsed_image_filenames": [],
        }

        if self.output_tos_path:
            output_name = (
                f"{Path(file_name_str).stem}.txt"
                if file_name_str
                else f"{int(time.time())}_{random.randint(1, 100000)}.txt"
            )
            try:
                markdown_final, _, tos_path = await self._async_transform_md(markdown, output_name)
                result["parsed_file_path"] = tos_path
                result["parsed_image_filenames"] = self._extract_image_filename(markdown_final)
                # Also persist detail JSON to TOS
                with tempfile.TemporaryDirectory(dir="/tmp") as temp_sub_dir:
                    detail_filename = f"{Path(output_name).stem}.detail.json"
                    tmp_detail_file = Path(temp_sub_dir) / detail_filename
                    with tmp_detail_file.open("w") as f:
                        json.dump(detail_list, f, ensure_ascii=False)
                    upload_file(str(tmp_detail_file), f"{self.output_detail_tos_path}/{detail_filename}")
            except Exception:
                logger.exception("Failed to transform and upload markdown or detail JSON")

        return markdown, result

    def _refresh_service(self) -> None:
        self._visual_service = get_visual_service(VisualServiceConfig.from_env())

    def _is_retryable_error(self, code: int | None) -> bool:
        """Classify retryable error codes according to service docs."""
        return code in {50429, 50500, 50501}

    def _summarize_req(self, req: dict[str, Any]) -> str:
        """Summarize request for logging without dumping large payloads."""
        try:
            src = "base64" if "image_base64" in req else "url" if "image_url" in req else "unknown"
            summary = {
                "src": src,
                "image_url": req.get("image_url"),  # log the pdf url if present to identify problematic files
                "page_start": req.get("page_start"),
                "page_num": req.get("page_num"),
                "parse_mode": req.get("parse_mode"),
                "table_mode": req.get("table_mode"),
                "version": req.get("version"),
                "file_type": req.get("file_type"),
            }
            return json.dumps(summary, ensure_ascii=False)
        except Exception:
            return "{log_summarize_error}"

    def transform(
        self,
        data_col: pa.Array,
        file_name_col: pa.Array | None = None,
    ) -> pa.Array:
        """根据 input_type 从 URL 或 Base64 编码的列中异步解析 PDF 文件。

        Args:
            data_col: 包含 PDF 文件 URL 或 Base64 编码内容的数组
            file_name_col: 包含文件名的数组，用于在 TOS 中保存

        Returns:
            pa.Array: 一个结构体数组，包含解析结果。
                - parsed_origin_text: 解析后的原始 Markdown 文本。
                - parsed_plain_text: 移除图片链接后的纯文本。
                - parsed_detail: JSON 格式的详细解析信息。
                - parsed_file_path: 解析结果在 TOS 上的存储路径。
                - parsed_image_filenames: 解析结果中图片文件名列表。
        """  # noqa: D415
        if self.input_type == "url":
            urls = data_col.to_pylist()
            base64s = [None] * len(urls)
        elif self.input_type == "base64":
            base64s = data_col.to_pylist()
            urls = [None] * len(base64s)
        else:
            raise ValueError(f"Invalid input_type: {self.input_type}")

        num_rows = len(data_col)
        file_names = file_name_col.to_pylist() if file_name_col is not None else [None] * num_rows

        async def process_row(index: int, url: str | None, b64: str | None, fname: str | None) -> dict[str, Any]:
            try:
                _, results = await self._process_pdf_source(b64, url, fname)
                return {
                    "parsed_origin_text": results.get("parsed_origin_text", ""),
                    "parsed_plain_text": results.get("parsed_plain_text", ""),
                    "parsed_detail": json.dumps(results.get("parsed_detail", []), ensure_ascii=False),
                    "parsed_file_path": results.get("parsed_file_path", ""),
                    "parsed_image_filenames": results.get("parsed_image_filenames", []),
                }
            except Exception:
                logger.exception("PDF processing failed for index %d", index)
                return {
                    "parsed_origin_text": "",
                    "parsed_plain_text": "",
                    "parsed_detail": "[]",
                    "parsed_file_path": "",
                    "parsed_image_filenames": [],
                }

        async def process_all_rows() -> list[dict[str, Any]]:
            tasks = [
                process_row(i, url, b64, fname) for i, (url, b64, fname) in enumerate(zip(urls, base64s, file_names))
            ]
            return await asyncio.gather(*tasks)

        results = asyncio.run(process_all_rows())

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.struct(
            [
                pa.field("parsed_origin_text", pa.string()),
                pa.field("parsed_plain_text", pa.string()),
                pa.field("parsed_detail", pa.string()),
                pa.field("parsed_file_path", pa.string()),
                pa.field("parsed_image_filenames", pa.list_(pa.string())),
            ]
        )
