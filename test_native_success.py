from __future__ import annotations

import os
import time

import daft
from daft import udf

# 配置 Dashboard 地址
if "DAFT_DASHBOARD_URL" not in os.environ:
    os.environ["DAFT_DASHBOARD_URL"] = "http://127.0.0.1:9000"

# 使用 Native 引擎
daft.set_runner_native()

print("Setting up a successful Native engine query...")


@udf(return_dtype=daft.DataType.int64())
def slow_inc(x):
    print("Native processing batch...")
    time.sleep(1)
    data = x.to_pylist()
    return [i + 1 for i in data]


# 创建 DataFrame
df = daft.from_pydict({"a": list(range(50))})
# Note: NativeRunner doesn't support repartitioning in the same way,
# but we can at least run a simple pipeline.
df = df.with_column("b", slow_inc(df["a"]))

print(f"Native Query starting at: {time.ctime()}")
print(f"Dashboard URL: {os.environ['DAFT_DASHBOARD_URL']}")

# 开始执行
try:
    result = df.collect()
    print(f"Native Query finished at: {time.ctime()}")
    print(f"Final row count: {len(result)}")
except Exception:
    print(f"Native Query failed at: {time.ctime()}")
    import traceback

    traceback.print_exc()
