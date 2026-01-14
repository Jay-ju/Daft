from __future__ import annotations

import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set up environment for Ray Runner and Dashboard
# Use the environment variable if set, otherwise default to the node's IP
if "DAFT_DASHBOARD_URL" not in os.environ:
    os.environ["DAFT_DASHBOARD_URL"] = "http://127.0.0.1:9000"

import daft

daft.refresh_logger()
import time

# Ensure we use Ray runner
daft.set_runner_ray(address="auto")

print("Running Daft Query on Ray...")

# Create a simple dataframe
df = daft.from_pydict({"a": list(range(1000))})

# Do some operations that might take a tiny bit of time or at least be tracked
df = df.with_column("b", df["a"] * 2)
df = df.with_column("c", df["a"] + df["b"])

# Repartition to trigger shuffle/stages if possible, though simple map might be fused
df = df.repartition(4)

# Collect results
print("Collecting results...")
result = df.collect()

print(f"Got {len(result)} rows")
print("Waiting for dashboard to finish...")
time.sleep(5)
print("Done!")
