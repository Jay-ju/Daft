from __future__ import annotations

import logging
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import tracking_usage
from daft.las.io import mkdirs, upload_file

logger = logging.getLogger(__name__)


class DocConvert(Operator):
    """**文档格式转换处理器，支持多种办公文档格式互转**

    **核心功能：**
    - 支持 doc/docx 到多种格式的转换
    - 使用 LibreOffice 进行高质量转换
    - 支持本地和云端存储路径
    - 提供转换超时控制
    - 自动处理文件上传下载

    **格式支持：**
    - 输入格式：Microsoft Word (.doc, .docx)
    - 输出格式：PDF (.pdf), ODT (.odt), HTML (.html), 纯文本 (.txt), DOCX (.docx)
    - 建议使用 docx 格式以获得最佳转换效果
    """  # noqa: D415

    def __init__(self, target_format: str, output_dir: str, convert_time_out: int = 60, **kwargs: Any) -> None:
        """Initialization.

        Args:
            target_format: 目标格式，支持 pdf/odt/html/txt/docx
            output_dir: 输出路径，必须提供，可以是本地路径或 TOS/S3 路径
            convert_time_out: 转换超时时间，单位为秒
        """
        super().__init__(**kwargs)
        if output_dir is None:
            raise ValueError("output_dir must be provided")

        self.output_dir = output_dir
        self.target_format = target_format
        self.convert_time_out = convert_time_out

        if shutil.which("soffice") is None:
            err_msg = """soffice command was not found. Please install libreoffice
                on your system and try again.

                - Install instructions: https://www.libreoffice.org/get-help/install-howto/
                - Mac: https://formulae.brew.sh/cask/libreoffice
                - Debian: https://wiki.debian.org/LibreOffice
            """
            raise ValueError(textwrap.dedent(err_msg))

        if output_dir:
            mkdirs(output_dir)

        tracking_usage(op=self.__class__.__name__, model_service_or_lib="soffice")

    def transform(self, files: pa.Array) -> pa.Array:
        """对文件进行格式转换.

        files:
            input: 输入文件路径，支持 tos 与 https 路径
        Returns:
            输出文件路径，仅支持 tos 路径
        """
        output_files = []

        for file in files:
            input_file = str(file)
            logger.info("Process files: %s started", input_file)
            output_file = self.convert_office_file(input_file)
            output_files.append(output_file)
            logger.info("Process file: %s finished", output_file)

        return pa.array(output_files, type=self.__return_column_type__())

    def convert_office_file(self, input_file: str) -> str:
        """Converts doc file using the libreoffice CLI.

        Args:
        input_file: str
            The input path of the file to be converted

        References:
            https://stackoverflow.com/questions/52277264/convert-doc-to-docx-using-soffice-not-working
            https://git.libreoffice.org/core/+/refs/heads/master/filter/source/config/fragments/filters
        """

        def convert_and_upload(local_file_path: str) -> str:
            """Convert a local file and upload the result."""
            input_file_path = Path(local_file_path)
            input_file_stem = input_file_path.stem
            tmp_dir = input_file_path.parent

            command = [
                "soffice",
                "--headless",
                "--convert-to",
                self.target_format,
                "--outdir",
                str(tmp_dir),
                local_file_path,
            ]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                process.wait(timeout=self.convert_time_out)
            except subprocess.TimeoutExpired:
                process.terminate()
                logger.error("Processing file %s timeout, return empty.", input_file)
                return ""
            except Exception as e:
                logger.error("Error during conversion of %s: %s", input_file, e)
                return ""

            output_file_name = f"{input_file_stem}.{self.target_format}"
            output_file_path = str(tmp_dir / output_file_name)

            if not Path(output_file_path).exists():
                logger.error("Converted file not found: %s", output_file_path)
                return ""

            output_file_final = f"{self.output_dir}/{output_file_name}"
            try:
                upload_file(output_file_path, output_file_final)
            except Exception as e:
                logger.error("Failed to upload converted file %s: %s", output_file_path, e)
                return ""

            return output_file_final

        try:
            from daft.las.functions.utils.common_utils import run_on_local_path

            return run_on_local_path(input_file, convert_and_upload)
        except Exception as e:
            logger.error("Failed to convert file %s: %s", input_file, e)
            return ""

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.string()
