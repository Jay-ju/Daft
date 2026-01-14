from __future__ import annotations

import os
import time

import daft
from daft import udf

# 配置 Dashboard 地址
if "DAFT_DASHBOARD_URL" not in os.environ:
    os.environ["DAFT_DASHBOARD_URL"] = "http://127.0.0.1:9000"

# 使用 Ray Runner (Flotilla Engine)
daft.set_runner_ray(address="auto")

print("Setting up a successful Flotilla (Ray) query...")


@udf(return_dtype=daft.DataType.int64())
def slow_inc(x):
    # x is a daft.Series
    print("Processing batch...")
    time.sleep(1)
    # Convert to list, increment, and return as list or Series
    data = x.to_pylist()
    return [i + 1 for i in data]


# 创建一个有多个分区的 DataFrame
df = daft.from_pydict({"a": list(range(200))})
df = df.repartition(5)

# 应用 UDF
df = df.with_column("b", slow_inc(df["a"]))

# 再做一次重分区
df = df.repartition(2)

print(f"Flotilla Query starting at: {time.ctime()}")
print(f"Dashboard URL: {os.environ['DAFT_DASHBOARD_URL']}")

# 开始执行
try:
    result = df.collect()
    print(f"Flotilla Query finished at: {time.ctime()}")
    print(f"Final row count: {len(result)}")
except Exception:
    print(f"Flotilla Query failed at: {time.ctime()}")
    import traceback

    traceback.print_exc()
