# Copyright (c) Beijing Volcano Engine Technology Ltd.

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import re
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import UnionType
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

import pyarrow as pa
from docstring_parser import parse
from function_meta.meta import (
    OP_BUCKET,
    OP_ENVIRONMENT,
    OP_VERSION,
    ExtraMetaModel,
    InputModel,
    OpMetaModel,
    OutputModel,
    ParameterModel,
)

from daft.las.io import upload_file

sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class ComposedModel:
    Name: str
    OperatorId: str
    Description: str
    FunctionCategory: str
    SubFunctionCategory: str
    Tags: list[str] | None = None
    Precondition: str | None = None
    Parameters: list[ParameterModel] = field(default_factory=list)
    Input: list[InputModel] = field(default_factory=list)
    Output: list[OutputModel] = field(default_factory=list)
    ExtraMeta: ExtraMetaModel | None = None


def _collect_all_metas(root_dir: str) -> dict[str, tuple[OpMetaModel, ExtraMetaModel]]:
    results = {}
    for py_file in Path(root_dir).rglob("*.py"):
        if py_file.name == "__init__.py" or not py_file.stem.endswith("_meta"):
            continue

        module_name = py_file.stem
        spec = importlib.util.spec_from_file_location(module_name, str(py_file))
        if spec is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if hasattr(module, "get_meta") and hasattr(module, "get_extra_meta"):
            op_meta = module.get_meta()
            op_meta.Description = op_meta.Clazz.__doc__
            results[module_name] = (op_meta, module.get_extra_meta())

    return results


def _collect_all_examples(root_dir: str) -> dict[str, str]:
    """Collect all example codes, and their corresponding tos dir."""
    results = {}
    for py_file in Path(root_dir).rglob("*.py"):
        if py_file.name == "__init__.py" or py_file.stem.endswith("_meta"):
            continue

        operator_id = py_file.stem
        target_dir = f"tos://{OP_BUCKET}/{OP_ENVIRONMENT}/operator_cards/{OP_VERSION}/{operator_id}/{operator_id}.py"
        results[str(py_file)] = target_dir

    return results


def _extract_precondition(clazz: type) -> str | None:
    doc = clazz.__doc__
    pattern = rf"{'Notes'}:\n(.*?)(?=\n\w+:|$)"
    match = re.search(pattern, doc, re.DOTALL)
    if match:
        # remove indent
        return textwrap.dedent(match.group(1)).strip()
    return None


def _get_type_name(type_hint):
    origin = get_origin(type_hint)
    if origin is Union or isinstance(type_hint, UnionType):
        args = get_args(type_hint)
        union_type = " or ".join(_get_type_name(arg) for arg in args)
        return _replace_with_alias(union_type)
    if hasattr(type_hint, "__name__"):
        return _replace_with_alias(type_hint.__name__)
    return _replace_with_alias(str(type_hint))


def _replace_with_alias(type_str: str):
    return type_str.replace("NoneType", "None")


def _extract_fn_info(fn: Callable) -> tuple[list[ParameterModel], OutputModel]:
    init_signature = inspect.signature(fn)

    parameters: list[dict[str, str]] = []
    for name, param in init_signature.parameters.items():
        if name == "self" or name == "args" or name == "kwargs":
            continue

        globalns = {"pa": pa}  # Ensure 'pa' is defined
        globalns.update(fn.__globals__)
        type_hint = get_type_hints(fn, globalns=globalns).get(name, None)
        type_hint = _get_type_name(type_hint)
        default = param.default if param.default != inspect.Parameter.empty else None

        parameters.append(
            {
                "name": name,
                "type": type_hint,
                "default": default,
            }
        )

    # parse the doc
    docstring = inspect.getdoc(fn)
    if not docstring:
        raise ValueError("Missing python doc.")

    parsed_doc = parse(docstring)
    for param_doc in parsed_doc.params:
        for param in parameters:
            if param["name"] == param_doc.arg_name:
                param["description"] = param_doc.description
                break
    return_desc = parsed_doc.returns.description if parsed_doc.returns is not None else None

    return [
        ParameterModel(
            Name=p.get("name"),
            Type=p.get("type", ""),
            Default=p.get("default", ""),
            Description=p.get("description", ""),
        )
        for p in parameters
    ], OutputModel(Description=return_desc)


def _extract_input_output(cls: type) -> tuple[list[InputModel], OutputModel]:
    inputs, output = _extract_fn_info(cls.transform)
    inputs = [
        InputModel(
            Name=input.Name,
            Description=input.Description,
        )
        for input in inputs
    ]
    return inputs, output


def _collect_ops() -> list[Any]:
    ops = _collect_all_metas(str(Path(__file__).parent / "function_meta"))

    result = []
    for name, meta in ops.items():
        op_meta = meta[0]
        extra_meta = meta[1]
        precondition = _extract_precondition(op_meta.Clazz)
        parameters, _ = _extract_fn_info(op_meta.Clazz.__init__)
        input, output = _extract_input_output(op_meta.Clazz)

        module = op_meta.Clazz.__module__
        qualname = op_meta.Clazz.__qualname__
        op = ComposedModel(
            Name=op_meta.Name,
            OperatorId=f"{module}.{qualname}",
            Description=op_meta.Description,
            FunctionCategory=op_meta.Category.value,
            SubFunctionCategory=op_meta.SubCategory.value,
            Tags=op_meta.Tags,
            Parameters=parameters,
            Input=input,
            Output=[output],
            Precondition=precondition,
            ExtraMeta=extra_meta,
        )
        result.append(asdict(op))
    return result


def _print_op_info(output_file: str = "ops.json"):
    res = json.dumps(_collect_ops(), ensure_ascii=False)
    with Path(output_file).open(mode="w", encoding="utf-8") as file:
        json.dump(json.loads(res), file, ensure_ascii=False, indent=4)
    print(res)


def _upload_examples():
    examples = _collect_all_examples(str(Path(__file__).parent / "function_meta"))
    for py_file, target_dir in examples.items():
        upload_file(str(py_file), target_dir)


if __name__ == "__main__":
    """
    usage: release_ops.py [-h] [-p/--print-ops] [-u/--upload-examples] [--env ENV]

    发布算子，支持打印算子信息和上传示例代码到 TOS。

    options:
      -h, --help            打印帮助信息
      -p, --print-ops       打印算子信息
      -u, --upload-examples 上传示例代码
    """

    parser = argparse.ArgumentParser(description="算子发布工具")
    parser.add_argument("--print-ops", "-p", action="store_true", help="打印算子信息")
    parser.add_argument("--upload-examples", "-u", action="store_true", help="上传示例代码")

    args = parser.parse_args()

    if not args.print_ops and not args.upload_examples:
        print("请至少选择一个操作: --print-ops 或 --upload-examples")
        parser.print_help()
        sys.exit(1)

    if args.print_ops:
        _print_op_info()

    if args.upload_examples:
        _upload_examples()
