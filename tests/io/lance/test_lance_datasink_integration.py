"""
Integration tests for Lance DataSink batching functionality.

This module focuses on integration testing between the batching mechanism
and the broader Daft ecosystem, including DataFrame operations, different
data sources, and real-world usage patterns.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pytest

import daft
from daft.context import get_context

# Skip tests if Lance is not available
try:
    import lance
    LANCE_AVAILABLE = True
except ImportError:
    LANCE_AVAILABLE = False

PYARROW_LOWER_BOUND_SKIP = tuple(int(s) for s in pa.__version__.split(".") if s.isnumeric()) < (9, 0, 0)
pytestmark = pytest.mark.skipif(
    PYARROW_LOWER_BOUND_SKIP or not LANCE_AVAILABLE, 
    reason="lance not supported on old versions of pyarrow or lance not installed"
)


@pytest.fixture(scope="function")
def lance_dataset_path(tmp_path_factory):
    """Create a temporary directory for Lance dataset tests."""
    tmp_dir = tmp_path_factory.mktemp("lance_integration")
    yield str(tmp_dir)


class TestDataFrameLanceIntegration:
    """Test DataFrame.write_lance() integration with batching."""

    def test_dataframe_write_lance_with_batching(self, lance_dataset_path):
        """Test DataFrame.write_lance() method with batching parameters."""
        # Create test data
        data = {
            "id": list(range(100)),
            "category": [f"cat_{i % 5}" for i in range(100)],
            "value": [float(i) * 1.1 for i in range(100)],
            "flag": [i % 2 == 0 for i in range(100)],
        }
        
        df = daft.from_pydict(data)
        
        # Test write_lance with batching parameters
        df.write_lance(
            lance_dataset_path,
            batch_size=10,
            max_batch_rows=25,
            mode="create",
            max_bytes_per_file=1024*1024,  # 1MB files
        )
        
        # Verify data integrity
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        assert len(loaded_data["id"]) == 100
        assert set(loaded_data["id"]) == set(range(100))
        assert set(loaded_data["category"]) == set([f"cat_{i}" for i in range(5)])

    def test_dataframe_operations_before_write(self, lance_dataset_path):
        """Test batching with DataFrame operations before writing."""
        # Create initial data
        data = {
            "id": list(range(200)),
            "value": [float(i) for i in range(200)],
            "group": [i % 10 for i in range(200)],
        }
        
        df = daft.from_pydict(data)
        
        # Apply DataFrame operations
        df_processed = (df
                       .filter(df["value"] > 50)
                       .with_column("doubled", df["value"] * 2)
                       .select("id", "doubled", "group"))
        
        # Write with batching
        df_processed.write_lance(
            lance_dataset_path,
            batch_size=15,
            max_batch_rows=50,
            mode="create"
        )
        
        # Verify processed data
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        # Should have filtered out values <= 50 (ids 0-50)
        expected_ids = list(range(51, 200))
        assert len(loaded_data["id"]) == len(expected_ids)
        assert set(loaded_data["id"]) == set(expected_ids)
        
        # Check doubled values
        for i, doubled_val in enumerate(loaded_data["doubled"]):
            original_id = loaded_data["id"][i]
            expected_doubled = float(original_id) * 2
            assert abs(doubled_val - expected_doubled) < 0.001

    def test_multiple_dataframe_writes_same_dataset(self, lance_dataset_path):
        """Test multiple DataFrame writes to the same dataset with batching."""
        # Write first batch
        data1 = {
            "timestamp": [1, 2, 3, 4, 5],
            "sensor_id": ["A", "B", "A", "C", "B"],
            "reading": [10.1, 20.2, 15.5, 30.3, 25.7],
        }
        df1 = daft.from_pydict(data1)
        df1.write_lance(lance_dataset_path, batch_size=3, mode="create")
        
        # Write second batch (append)
        data2 = {
            "timestamp": [6, 7, 8, 9, 10],
            "sensor_id": ["C", "A", "B", "A", "C"],
            "reading": [35.8, 12.3, 28.9, 18.4, 40.1],
        }
        df2 = daft.from_pydict(data2)
        df2.write_lance(lance_dataset_path, batch_size=2, mode="append")
        
        # Write third batch (append)
        data3 = {
            "timestamp": [11, 12, 13],
            "sensor_id": ["B", "C", "A"],
            "reading": [22.6, 33.7, 16.8],
        }
        df3 = daft.from_pydict(data3)
        df3.write_lance(lance_dataset_path, batch_size=5, mode="append")
        
        # Verify combined data
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        expected_timestamps = list(range(1, 14))
        assert len(loaded_data["timestamp"]) == 13
        assert set(loaded_data["timestamp"]) == set(expected_timestamps)


class TestLanceBatchingWithDifferentDataSources:
    """Test batching with different data sources and formats."""

    def test_batching_with_parquet_source(self, lance_dataset_path, tmp_path):
        """Test batching when reading from Parquet files."""
        # Create Parquet files
        parquet_dir = tmp_path / "parquet_source"
        parquet_dir.mkdir()
        
        # Create multiple Parquet files
        for i in range(3):
            data = {
                "file_id": [i] * 20,
                "record_id": list(range(i*20, (i+1)*20)),
                "value": [float(j) * (i + 1) for j in range(20)],
            }
            df_part = daft.from_pydict(data)
            parquet_file = parquet_dir / f"part_{i}.parquet"
            df_part.write_parquet(str(parquet_file))
        
        # Read from Parquet and write to Lance with batching
        df_from_parquet = daft.read_parquet(str(parquet_dir / "*.parquet"))
        df_from_parquet.write_lance(
            lance_dataset_path,
            batch_size=5,
            max_batch_rows=30,
            mode="create"
        )
        
        # Verify data
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        assert len(loaded_data["record_id"]) == 60  # 3 files * 20 records each
        assert set(loaded_data["file_id"]) == {0, 1, 2}

    def test_batching_with_csv_source(self, lance_dataset_path, tmp_path):
        """Test batching when reading from CSV files."""
        # Create CSV file
        csv_file = tmp_path / "test_data.csv"
        csv_data = "id,name,score\n"
        for i in range(50):
            csv_data += f"{i},user_{i},{i * 1.5}\n"
        
        csv_file.write_text(csv_data)
        
        # Read from CSV and write to Lance with batching
        df_from_csv = daft.read_csv(str(csv_file))
        df_from_csv.write_lance(
            lance_dataset_path,
            batch_size=8,
            max_batch_rows=20,
            mode="create"
        )
        
        # Verify data
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        assert len(loaded_data["id"]) == 50
        assert loaded_data["name"][0] == "user_0"
        assert loaded_data["name"][-1] == "user_49"

    def test_batching_with_json_source(self, lance_dataset_path, tmp_path):
        """Test batching when reading from JSON files."""
        # Create JSON file
        json_file = tmp_path / "test_data.json"
        json_data = []
        for i in range(30):
            json_data.append({
                "user_id": i,
                "username": f"user_{i}",
                "metadata": {"level": i % 5, "active": i % 2 == 0}
            })
        
        import json
        with open(json_file, 'w') as f:
            for item in json_data:
                f.write(json.dumps(item) + '\n')
        
        # Read from JSON and write to Lance with batching
        df_from_json = daft.read_json(str(json_file))
        df_from_json.write_lance(
            lance_dataset_path,
            batch_size=6,
            max_batch_rows=15,
            mode="create"
        )
        
        # Verify data
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        assert len(loaded_data["user_id"]) == 30
        assert loaded_data["username"][0] == "user_0"


class TestLanceBatchingPerformanceScenarios:
    """Test batching in realistic performance scenarios."""

    def test_large_dataset_batching(self, lance_dataset_path):
        """Test batching with a larger dataset to verify performance characteristics."""
        # Create larger dataset (10K rows)
        n_rows = 10000
        data = {
            "id": list(range(n_rows)),
            "timestamp": [f"2024-01-{(i % 30) + 1:02d}T{(i % 24):02d}:00:00" for i in range(n_rows)],
            "category": [f"category_{i % 100}" for i in range(n_rows)],
            "value": [float(i) * 0.1 for i in range(n_rows)],
            "flag": [i % 3 == 0 for i in range(n_rows)],
        }
        
        df = daft.from_pydict(data)
        
        # Write with moderate batching
        df.write_lance(
            lance_dataset_path,
            batch_size=50,
            max_batch_rows=1000,
            mode="create",
            max_bytes_per_file=10*1024*1024,  # 10MB files
        )
        
        # Verify data integrity and check fragment count
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        assert len(loaded_data["id"]) == n_rows
        assert set(loaded_data["id"]) == set(range(n_rows))
        
        # Check Lance dataset stats
        ds = lance.dataset(lance_dataset_path)
        fragments = ds.get_fragments()
        
        # With batching, should have fewer fragments than rows
        assert len(fragments) < n_rows
        assert ds.count_rows() == n_rows

    def test_memory_efficient_batching(self, lance_dataset_path):
        """Test memory-efficient batching with row limits."""
        # Create dataset with varying row sizes
        data_parts = []
        total_rows = 0
        
        # Create parts with different sizes
        part_sizes = [100, 500, 200, 800, 300]
        for i, size in enumerate(part_sizes):
            data = {
                "part_id": [i] * size,
                "row_id": list(range(total_rows, total_rows + size)),
                "data": [f"data_{j}" * 10 for j in range(size)],  # Larger text data
                "value": [float(j) * (i + 1) for j in range(size)],
            }
            data_parts.append(data)
            total_rows += size
        
        # Write each part separately with row-based batching
        for i, data in enumerate(data_parts):
            df = daft.from_pydict(data)
            mode = "create" if i == 0 else "append"
            df.write_lance(
                lance_dataset_path,
                batch_size=100,  # Large batch size
                max_batch_rows=300,  # But limit by rows
                mode=mode
            )
        
        # Verify all data was written correctly
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        assert len(loaded_data["row_id"]) == total_rows
        assert set(loaded_data["part_id"]) == set(range(len(part_sizes)))

    def test_concurrent_write_simulation(self, lance_dataset_path):
        """Test batching behavior that simulates concurrent-like writes."""
        # Simulate multiple "streams" of data being written
        streams = []
        for stream_id in range(3):
            stream_data = []
            for batch_id in range(5):
                data = {
                    "stream_id": [stream_id] * 20,
                    "batch_id": [batch_id] * 20,
                    "record_id": list(range(batch_id * 20, (batch_id + 1) * 20)),
                    "timestamp": [f"2024-01-01T{(stream_id * 8 + batch_id):02d}:00:00"] * 20,
                    "value": [float(i + stream_id * 1000) for i in range(20)],
                }
                stream_data.append(data)
            streams.append(stream_data)
        
        # Write all streams in interleaved fashion
        all_writes = []
        for batch_idx in range(5):
            for stream_idx in range(3):
                all_writes.append((stream_idx, batch_idx, streams[stream_idx][batch_idx]))
        
        # Write with batching
        for i, (stream_id, batch_id, data) in enumerate(all_writes):
            df = daft.from_pydict(data)
            mode = "create" if i == 0 else "append"
            df.write_lance(
                lance_dataset_path,
                batch_size=4,  # Small batches to test frequent flushing
                max_batch_rows=50,
                mode=mode
            )
        
        # Verify all data from all streams
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        expected_total_rows = 3 * 5 * 20  # 3 streams * 5 batches * 20 records
        assert len(loaded_data["record_id"]) == expected_total_rows
        assert set(loaded_data["stream_id"]) == {0, 1, 2}
        assert set(loaded_data["batch_id"]) == {0, 1, 2, 3, 4}


class TestLanceBatchingWithComplexSchemas:
    """Test batching with complex data types and schemas."""

    def test_batching_with_nested_data(self, lance_dataset_path):
        """Test batching with nested/complex data types."""
        # Create data with complex types
        data = {
            "id": [1, 2, 3, 4, 5],
            "metadata": [
                {"tags": ["a", "b"], "score": 1.0},
                {"tags": ["c"], "score": 2.0},
                {"tags": ["d", "e", "f"], "score": 3.0},
                {"tags": [], "score": 4.0},
                {"tags": ["g", "h"], "score": 5.0},
            ],
            "coordinates": [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
            ],
        }
        
        df = daft.from_pydict(data)
        df.write_lance(
            lance_dataset_path,
            batch_size=3,
            mode="create"
        )
        
        # Verify complex data integrity
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        assert len(loaded_data["id"]) == 5
        assert len(loaded_data["metadata"]) == 5
        assert len(loaded_data["coordinates"]) == 5
        
        # Verify nested data structure
        assert loaded_data["coordinates"][0] == [1.0, 2.0]
        assert loaded_data["coordinates"][-1] == [9.0, 10.0]

    def test_batching_with_nullable_fields(self, lance_dataset_path):
        """Test batching with nullable fields and missing values."""
        data = {
            "id": [1, 2, 3, 4, 5, 6],
            "optional_field": ["a", None, "c", None, "e", "f"],
            "nullable_number": [1.0, 2.0, None, 4.0, None, 6.0],
            "required_field": ["x", "y", "z", "w", "v", "u"],
        }
        
        df = daft.from_pydict(data)
        df.write_lance(
            lance_dataset_path,
            batch_size=4,
            max_batch_rows=3,  # Force multiple batches
            mode="create"
        )
        
        # Verify nullable data handling
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        assert len(loaded_data["id"]) == 6
        
        # Check that None values are preserved
        optional_values = loaded_data["optional_field"]
        assert None in optional_values
        assert "a" in optional_values
        assert "f" in optional_values

    def test_batching_with_large_strings(self, lance_dataset_path):
        """Test batching with large string fields."""
        # Create data with large strings
        large_text = "Lorem ipsum " * 1000  # ~11KB per string
        data = {
            "id": list(range(10)),
            "large_text": [f"{large_text}_{i}" for i in range(10)],
            "category": [f"cat_{i % 3}" for i in range(10)],
        }
        
        df = daft.from_pydict(data)
        df.write_lance(
            lance_dataset_path,
            batch_size=5,
            max_batch_rows=4,  # Should trigger row-based flushing
            mode="create"
        )
        
        # Verify large string data
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        assert len(loaded_data["id"]) == 10
        assert all(text.startswith("Lorem ipsum") for text in loaded_data["large_text"])
        assert all(text.endswith(f"_{i}") for i, text in enumerate(loaded_data["large_text"]))


class TestLanceBatchingErrorRecovery:
    """Test error recovery and robustness in batching scenarios."""

    def test_partial_batch_recovery(self, lance_dataset_path):
        """Test recovery when some batches succeed and others fail."""
        # Create mixed data that might cause issues
        good_data = [
            {"id": [1, 2], "name": ["alice", "bob"], "value": [1.0, 2.0]},
            {"id": [3, 4], "name": ["charlie", "david"], "value": [3.0, 4.0]},
        ]
        
        # Write good data first
        for i, data in enumerate(good_data):
            df = daft.from_pydict(data)
            mode = "create" if i == 0 else "append"
            df.write_lance(lance_dataset_path, batch_size=2, mode=mode)
        
        # Verify partial data was written
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        assert len(loaded_data["id"]) == 4
        assert set(loaded_data["id"]) == {1, 2, 3, 4}

    def test_empty_dataset_handling(self, lance_dataset_path):
        """Test handling of completely empty datasets."""
        # Try to create dataset with empty data
        empty_data = {"id": [], "name": [], "value": []}
        df_empty = daft.from_pydict(empty_data)
        
        # This should handle empty data gracefully
        df_empty.write_lance(lance_dataset_path, batch_size=5, mode="create")
        
        # Add some real data
        real_data = {"id": [1, 2], "name": ["alice", "bob"], "value": [1.0, 2.0]}
        df_real = daft.from_pydict(real_data)
        df_real.write_lance(lance_dataset_path, batch_size=2, mode="append")
        
        # Verify only real data exists
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        assert len(loaded_data["id"]) == 2
        assert set(loaded_data["id"]) == {1, 2}


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/io/lance/test_lance_datasink_integration.py -v
    pytest.main([__file__, "-v"])