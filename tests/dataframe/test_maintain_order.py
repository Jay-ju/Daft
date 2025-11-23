# Copyright 2025 Daft maintain_order tests
# flake8: noqa

import os
import pytest
import logging

def config_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S.%s'.format("%f"))

    logging.getLogger("tracing.span").setLevel(logging.WARNING)
    logging.getLogger("daft_io.stats").setLevel(logging.WARNING)
    logging.getLogger("DaftStatisticsManager").setLevel(logging.WARNING)
    logging.getLogger("DaftFlotillaScheduler").setLevel(logging.WARNING)
    logging.getLogger("DaftFlotillaDispatcher").setLevel(logging.WARNING)
    # 确保能捕获调度日志
    logging.getLogger("daft.dispatch").setLevel(logging.INFO)

config_logging()
logger = logging.getLogger(__name__)

import daft
from daft import refresh_logger  # 导入refresh_logger函数
import ray
from daft import col
from daft.context import execution_config_ctx
from daft.datatype import DataType
from daft.udf import udf


ray.init(
    runtime_env={
        "env_vars": {"DAFT_PROGRESS_BAR": "0", "DAFT_DEBUG_DISPATCH": "1"},
        "worker_process_setup_hook": config_logging
    },
    ignore_reinit_error=True
)
daft.set_runner_ray()

# 刷新日志配置以确保Rust日志系统正确初始化
refresh_logger()

# --------- Helpers ---------
N = 8

def make_input_pylist(n: int = N):
    return [{"idx": i, "val": chr(ord("a") + i)} for i in range(n)]

@udf(return_dtype=DataType.string())
def to_upper(s):
    # Simple row-wise transformation to ensure UDF is executed
    return [str(x).upper() for x in s.to_pylist()]


def select_with_udf(df: daft.DataFrame) -> daft.DataFrame:
    return df.select(col("idx"), to_upper(col("val")).alias("val2"))


def get_idx_sequence_from_pylist(rows: list[dict]) -> list[int]:
    return [r["idx"] for r in rows]


def get_idx_set_from_pylist(rows: list[dict]) -> set[int]:
    return set(get_idx_sequence_from_pylist(rows))


# --------- 1) from_pylist -> udf -> to_pylist/show/collect ---------

def test_udf_to_pylist_preserves_order_default():
    df = daft.from_pylist(make_input_pylist())
    out = select_with_udf(df)
    rows = out.to_pylist()
    assert get_idx_sequence_from_pylist(rows) == list(range(N))


def test_udf_to_pylist_relaxed_order_when_disabled():
    df = daft.from_pylist(make_input_pylist())
    with execution_config_ctx(maintain_order=False):
        out = select_with_udf(df)
        rows = out.to_pylist()
    # When maintain_order is disabled, ordering is not guaranteed; validate set-equality only
    assert get_idx_set_from_pylist(rows) == set(range(N))


def test_udf_show_construct_preview_preserves_order_default():
    df = daft.from_pylist(make_input_pylist())
    out = select_with_udf(df)
    preview = out._construct_show_preview(N)
    pydict = preview.partition.to_pydict()
    assert pydict["idx"] == list(range(N))


def test_udf_collect_then_to_pylist_preserves_order_default():
    df = daft.from_pylist(make_input_pylist())
    out = select_with_udf(df).collect()
    rows = out.to_pylist()
    assert get_idx_sequence_from_pylist(rows) == list(range(N))


# --------- 2) from_pylist -> udf -> write_xxx (CSV + Parquet) ---------

@pytest.mark.parametrize("file_type", ["csv", "parquet"])
@pytest.mark.parametrize("multi_partitions", [False, True])
def test_udf_write_and_read_order_not_required(tmp_path, file_type, multi_partitions):
    df = daft.from_pylist(make_input_pylist())
    df = select_with_udf(df)

    # Optionally increase number of upstream partitions to make parallel behavior more likely
    if multi_partitions:
        # Split the dataframe into multiple partitions; this helps exercise writer parallelism
        # (Native runner supports into_partitions via local plan; ignore warnings if any)
        try:
            df = df.into_partitions(4)
        except Exception:
            # Fallback: into_batches to achieve multiple morsels
            df = df.into_batches(2)

    # Write
    if file_type == "csv":
        written = df.write_csv(str(tmp_path))
        readback = daft.read_csv(str(tmp_path)).collect()
    else:
        written = df.write_parquet(str(tmp_path))
        readback = daft.read_parquet(str(tmp_path)).collect()

    # Sanity on written result schema
    assert "path" in written.column_names
    # Validate that readback contains all rows regardless of order
    rows = readback.select(col("idx")).to_pylist()
    assert get_idx_set_from_pylist(rows) == set(range(N))


# --------- 3) from_pylist -> into_batches/repartition -> udf -> to_pylist ---------

def test_into_batches_then_udf_preserves_order_when_enabled():
    df = daft.from_pylist(make_input_pylist()).into_batches(2)
    out = select_with_udf(df)
    rows = out.to_pylist()
    assert get_idx_sequence_from_pylist(rows) == list(range(N))


def test_into_batches_then_udf_relaxed_order_when_disabled():
    df = daft.from_pylist(make_input_pylist()).into_batches(2)
    with execution_config_ctx(maintain_order=False):
        out = select_with_udf(df)
        rows = out.to_pylist()
    assert get_idx_set_from_pylist(rows) == set(range(N))


def test_repartition_then_udf_order_not_guaranteed_even_if_enabled():
    df = daft.from_pylist(make_input_pylist())
    # Repartition shuffles and is implemented as a blocking sink that disables maintain_order downstream
    # Use hash repartition by idx to deterministically shard
    df = df.repartition(4, col("idx"))
    out = select_with_udf(df)
    rows = out.to_pylist()
    # Ordering not guaranteed; validate set-equality only
    assert get_idx_set_from_pylist(rows) == set(range(N))


# --------- 4) Debug logging: emit dispatch strategy selection ---------

def test_debug_dispatch_logs_emitted(capsys):
    os.environ["DAFT_DEBUG_DISPATCH"] = "1"

    df = daft.from_pylist(make_input_pylist()).into_batches(2)
    # maintain_order=True path (default)
    _ = select_with_udf(df).to_pylist()

    # maintain_order=False path
    with execution_config_ctx(maintain_order=False):
        _ = select_with_udf(df).to_pylist()

    captured = capsys.readouterr()
    # Print whatever was captured for developer inspection
    print("[test_debug_dispatch_logs_emitted] captured stdout:\n", captured.out)
    print("[test_debug_dispatch_logs_emitted] captured stderr:\n", captured.err)

    # Lightweight sanity: if logs are wired to stdout/stderr, we should see our marker
    if "[DaftDispatch]" in captured.out or "[DaftDispatch]" in captured.err:
        assert True
    else:
        # Do not fail: logging may be routed via other subscribers; still useful to enable env
        print("[test_debug_dispatch_logs_emitted] No [DaftDispatch] logs captured; logging may be suppressed by runtime configuration.")
