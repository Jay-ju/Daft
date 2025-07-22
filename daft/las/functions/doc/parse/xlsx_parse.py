# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd  # noqa: TID253

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import run_on_local_path, upload_folder
from daft.las.io import mkdirs

logger = logging.getLogger(__name__)


class XlsxParse(Operator):
    """**Excel 表格解析处理器，支持多格式输出与结构化数据提取**

    **核心功能：**
    - 支持 xlsx/xls 格式解析
    - 输出 markdown 或 html 格式
    - 保留表格结构与数据关系
    - 支持多工作表处理
    - 提供 TOS 存储选项

    **格式支持：**
    - Microsoft Excel (.xlsx, .xls)
    - 建议使用 xlsx 格式以获得最佳解析效果
    """  # noqa: D415

    def __init__(
        self,
        if_save_md_content: bool = True,
        if_save_html_content: bool = False,
        output_tos_path: str = "",
    ):
        """初始化 Xlsx 表格解析处理器

        Args:
            if_save_md_content: 是否保存为 markdown 格式，默认 True
            if_save_html_content: 是否保存为 html 格式，默认 False
                当两者都为 True 时，仅保存 markdown
            output_tos_path: 解析内容存储到的 TOS 目录,为空则不上传
        """  # noqa: D415
        super().__init__()
        if not if_save_md_content and not if_save_html_content:
            raise ValueError("if_save_md_content 和 if_save_html_content 至少有一个为 True")
        self.if_save_md_content = if_save_md_content
        self.if_save_html_content = if_save_html_content

        if isinstance(output_tos_path, str) and output_tos_path.startswith("tos://"):
            mkdirs(output_tos_path)
        else:
            raise ValueError("Invalid output_tos_path: it should be a valid tos path")
        self.output_tos_path = output_tos_path

    def transform(
        self,
        xlsx_col: pa.Array,
    ) -> pa.Array:
        """解析 Excel 文件，输出 markdown/html 文本及表格粒度文本

        Args:
            xlsx_col: 包含 xlsx/xls 文件路径的列

        Returns:
            struct 数组，字段包括
                - data_item_uri: 原始文件路径
                - text: 合并后的 markdown/html 文本
                - text_by_table: 每个 sheet 的 markdown/html 文本列表
        """  # noqa: D415
        input_files = xlsx_col.to_pylist()
        texts = []
        texts_by_table = []
        data_item_uris = []

        for file in input_files:
            logger.info("Processing file: %s", file)

            try:
                html_text_list, md_text_list = run_on_local_path(file, self._process_single_local_file)
                data_item_uris.append(file)

                if self.if_save_md_content:
                    texts.append("\n".join(md_text_list))
                    texts_by_table.append(md_text_list)
                else:
                    texts.append("\n".join(html_text_list))
                    texts_by_table.append(html_text_list)

            except Exception:
                logger.exception("Failed to process file %s", file)
                data_item_uris.append(file)
                texts.append("")
                texts_by_table.append([])

        result_structs = [
            {
                "data_item_uri": uri,
                "text": text,
                "text_by_table": tbls,
            }
            for uri, text, tbls in zip(data_item_uris, texts, texts_by_table)
        ]
        return pa.array(result_structs, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.struct(
            [
                pa.field("data_item_uri", pa.string()),
                pa.field("text", pa.string()),
                pa.field("text_by_table", pa.list_(pa.string())),
            ]
        )

    def _process_single_local_file(self, xlsx_path: str) -> tuple[list[str], list[str]]:
        """Parse local xlsx/xls file and output html/markdown text list.

        Args:
            xlsx_path (str): Local path to xlsx/xls file.

        Returns:
            tuple[list[str], list[str]]: (html_text_list, md_text_list)
        """
        xlsx_name = Path(xlsx_path).stem

        temp_dir = Path(xlsx_path).parent
        logger.info("Temp dir: %s", temp_dir)

        output_path = Path(temp_dir) / Path(xlsx_name)
        Path(output_path).mkdir(exist_ok=True)
        logger.info("The output local path: %s", output_path)

        html_output_path = output_path / Path("html")
        Path(html_output_path).mkdir(exist_ok=True)

        md_output_path = output_path / Path("md")
        Path(md_output_path).mkdir(exist_ok=True)

        try:
            xlsx_df = pd.read_excel(xlsx_path, sheet_name=None, header=0)
            html_text_list = []
            md_text_list = []
            xlsx_df_len = len(xlsx_df)
            for page_number, (sheet_name, sheet) in enumerate(xlsx_df.items(), start=1):
                logger.info("Processing page %d of %d", page_number, xlsx_df_len)
                html_text = sheet.to_html(index=False, header=True, na_rep="")
                sheet = sheet.fillna("")
                md_text = sheet.to_markdown(index=False)
                html_text_list.append(html_text)
                md_text_list.append(md_text)
                if self.if_save_html_content:
                    html_path = html_output_path / Path(f"{sheet_name}.html")
                    with Path.open(html_path, "w") as image_file:
                        image_file.write(html_text)

                if self.if_save_md_content:
                    md_path = md_output_path / Path(f"{sheet_name}.md")
                    with Path.open(md_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_text)

            if self.output_tos_path:
                upload_folder(str(output_path), self.output_tos_path)

        finally:
            shutil.rmtree(output_path)

        return html_text_list, md_text_list
