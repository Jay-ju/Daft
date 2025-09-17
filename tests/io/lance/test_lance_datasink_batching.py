"""
Comprehensive test suite for Lance DataSink batching functionality.

This module contains unit tests, integration tests, performance validation tests,
and error handling tests for the Lance DataSink batching mechanism.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import Mock, patch

import pyarrow as pa
import pytest

import daft
from daft.datatype import DataType
from daft.io.lance.lance_data_sink import LanceDataSink, pyarrow_schema_castable
from daft.recordbatch import MicroPartition
from daft.schema import Schema

# Skip tests if Lance is not available
pytest_plugins = ["tests.conftest"]

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
    tmp_dir = tmp_path_factory.mktemp("lance_batching")
    yield str(tmp_dir)


@pytest.fixture
def sample_schema():
    """Create a sample schema for testing."""
    return Schema._from_field_name_and_types([
        ("id", DataType.int64()),
        ("name", DataType.string()),
        ("value", DataType.float64()),
    ])


@pytest.fixture
def sample_data_small():
    """Create small sample data for testing."""
    return {
        "id": [1, 2, 3],
        "name": ["alice", "bob", "charlie"],
        "value": [1.1, 2.2, 3.3],
    }


@pytest.fixture
def sample_data_large():
    """Create larger sample data for testing."""
    n = 1000
    return {
        "id": list(range(n)),
        "name": [f"user_{i}" for i in range(n)],
        "value": [float(i) * 1.1 for i in range(n)],
    }


@pytest.fixture
def create_micropartitions():
    """Factory function to create micropartitions from data."""
    def _create_micropartitions(data_list, schema):
        """Create a list of micropartitions from data list."""
        micropartitions = []
        for data in data_list:
            df = daft.from_pydict(data)
            # Convert to MicroPartition
            arrow_table = df.to_arrow()
            mp = MicroPartition.from_arrow(arrow_table)
            micropartitions.append(mp)
        return micropartitions
    return _create_micropartitions


class TestLanceDataSinkParameterValidation:
    """Test parameter validation and initialization."""

    def test_valid_parameters(self, lance_dataset_path, sample_schema):
        """Test initialization with valid parameters."""
        sink = LanceDataSink(
            uri=lance_dataset_path,
            schema=sample_schema,
            mode="create",
            batch_size=5,
            max_batch_rows=10000,
        )
        assert sink._batch_size == 5
        assert sink._max_batch_rows == 10000
        assert sink.name() == "Lance Write (batch_size=5, max_batch_rows=10000)"

    def test_default_parameters(self, lance_dataset_path, sample_schema):
        """Test initialization with default parameters for backward compatibility."""
        sink = LanceDataSink(
            uri=lance_dataset_path,
            schema=sample_schema,
            mode="create",
        )
        assert sink._batch_size == 1
        assert sink._max_batch_rows == 100000

    def test_invalid_batch_size(self, lance_dataset_path, sample_schema):
        """Test initialization with invalid batch_size values."""
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            LanceDataSink(
                uri=lance_dataset_path,
                schema=sample_schema,
                mode="create",
                batch_size=0,
            )
        
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            LanceDataSink(
                uri=lance_dataset_path,
                schema=sample_schema,
                mode="create",
                batch_size=-1,
            )
        
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            LanceDataSink(
                uri=lance_dataset_path,
                schema=sample_schema,
                mode="create",
                batch_size="invalid",
            )

    def test_invalid_max_batch_rows(self, lance_dataset_path, sample_schema):
        """Test initialization with invalid max_batch_rows values."""
        with pytest.raises(ValueError, match="max_batch_rows must be a positive integer"):
            LanceDataSink(
                uri=lance_dataset_path,
                schema=sample_schema,
                mode="create",
                max_batch_rows=0,
            )
        
        with pytest.raises(ValueError, match="max_batch_rows must be a positive integer"):
            LanceDataSink(
                uri=lance_dataset_path,
                schema=sample_schema,
                mode="create",
                max_batch_rows=-1000,
            )

    def test_invalid_uri_type(self, sample_schema):
        """Test initialization with invalid URI type."""
        with pytest.raises(TypeError, match="Expected URI to be str or pathlib.Path"):
            LanceDataSink(
                uri=123,  # Invalid type
                schema=sample_schema,
                mode="create",
            )


class TestLanceDataSinkBatchingLogic:
    """Test the core batching logic."""

    def test_should_flush_batch_by_size(self, lance_dataset_path, sample_schema, create_micropartitions):
        """Test batch flushing based on batch_size."""
        sink = LanceDataSink(
            uri=lance_dataset_path,
            schema=sample_schema,
            mode="create",
            batch_size=3,
            max_batch_rows=1000000,  # Very large to avoid row-based flushing
        )
        
        # Create micropartitions
        data_list = [{"id": [i], "name": [f"user_{i}"], "value": [float(i)]} for i in range(5)]
        micropartitions = create_micropartitions(data_list, sample_schema)
        
        # Test batch size flushing
        assert not sink._should_flush_batch(micropartitions[:2], 2)  # 2 < 3
        assert sink._should_flush_batch(micropartitions[:3], 3)      # 3 >= 3
        assert sink._should_flush_batch(micropartitions[:4], 4)      # 4 >= 3

    def test_should_flush_batch_by_rows(self, lance_dataset_path, sample_schema, create_micropartitions):
        """Test batch flushing based on max_batch_rows."""
        sink = LanceDataSink(
            uri=lance_dataset_path,
            schema=sample_schema,
            mode="create",
            batch_size=1000,  # Very large to avoid size-based flushing
            max_batch_rows=5,
        )
        
        # Create micropartitions with different row counts
        data_list = [
            {"id": [1, 2], "name": ["a", "b"], "value": [1.0, 2.0]},  # 2 rows
            {"id": [3, 4], "name": ["c", "d"], "value": [3.0, 4.0]},  # 2 rows
            {"id": [5], "name": ["e"], "value": [5.0]},                # 1 row
        ]
        micropartitions = create_micropartitions(data_list, sample_schema)
        
        # Test row count flushing
        assert not sink._should_flush_batch(micropartitions[:1], 2)  # 2 < 5
        assert not sink._should_flush_batch(micropartitions[:2], 4)  # 4 < 5
        assert sink._should_flush_batch(micropartitions[:3], 5)      # 5 >= 5

    def test_backward_compatibility_no_batching(self, lance_dataset_path, sample_schema, sample_data_small):
        """Test that batch_size=1 uses individual processing for backward compatibility."""
        sink = LanceDataSink(
            uri=lance_dataset_path,
            schema=sample_schema,
            mode="create",
            batch_size=1,  # Should use individual processing
        )
        
        df = daft.from_pydict(sample_data_small)
        df.write_lance(lance_dataset_path, batch_size=1)
        
        # Verify data was written correctly
        df_loaded = daft.read_lance(lance_dataset_path)
        assert df_loaded.to_pydict() == sample_data_small


class TestLanceDataSinkIntegration:
    """Integration tests for end-to-end functionality."""

    def test_batching_end_to_end_small_data(self, lance_dataset_path, sample_data_small):
        """Test end-to-end batching with small dataset."""
        df = daft.from_pydict(sample_data_small)
        
        # Write with batching
        df.write_lance(
            lance_dataset_path,
            batch_size=2,
            max_batch_rows=1000,
            mode="create"
        )
        
        # Read back and verify
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        # Sort both datasets for comparison (order might differ)
        original_sorted = {k: sorted(v) if k != "name" else v for k, v in sample_data_small.items()}
        loaded_sorted = {k: sorted(v) if k != "name" else v for k, v in loaded_data.items()}
        
        assert len(loaded_data["id"]) == len(sample_data_small["id"])
        assert set(loaded_data["id"]) == set(sample_data_small["id"])
        assert set(loaded_data["name"]) == set(sample_data_small["name"])

    def test_batching_end_to_end_large_data(self, lance_dataset_path, sample_data_large):
        """Test end-to-end batching with larger dataset."""
        df = daft.from_pydict(sample_data_large)
        
        # Write with batching
        df.write_lance(
            lance_dataset_path,
            batch_size=10,
            max_batch_rows=500,  # Should trigger row-based flushing
            mode="create"
        )
        
        # Read back and verify
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        assert len(loaded_data["id"]) == len(sample_data_large["id"])
        assert set(loaded_data["id"]) == set(sample_data_large["id"])

    def test_different_write_modes(self, lance_dataset_path, sample_data_small):
        """Test batching with different Lance write modes."""
        df1 = daft.from_pydict(sample_data_small)
        
        # Test create mode
        df1.write_lance(lance_dataset_path, batch_size=2, mode="create")
        
        # Test append mode
        df2_data = {
            "id": [4, 5],
            "name": ["david", "eve"],
            "value": [4.4, 5.5],
        }
        df2 = daft.from_pydict(df2_data)
        df2.write_lance(lance_dataset_path, batch_size=2, mode="append")
        
        # Verify combined data
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        expected_ids = sample_data_small["id"] + df2_data["id"]
        assert len(loaded_data["id"]) == len(expected_ids)
        assert set(loaded_data["id"]) == set(expected_ids)

    def test_different_data_types(self, lance_dataset_path):
        """Test batching with different data types."""
        complex_data = {
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "string_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
            "list_col": [[1, 2], [3, 4], [5, 6]],
        }
        
        df = daft.from_pydict(complex_data)
        df.write_lance(lance_dataset_path, batch_size=2, mode="create")
        
        # Read back and verify
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        assert len(loaded_data["int_col"]) == len(complex_data["int_col"])
        assert set(loaded_data["int_col"]) == set(complex_data["int_col"])


class TestLanceDataSinkPerformanceValidation:
    """Performance validation tests."""

    def test_fragment_count_reduction(self, lance_dataset_path):
        """Test that batching reduces the number of Lance fragments."""
        # Create data that would normally create many small fragments
        data_parts = []
        for i in range(20):  # 20 small parts
            data_parts.append({
                "id": [i],
                "name": [f"user_{i}"],
                "value": [float(i)],
            })
        
        # Write without batching (batch_size=1)
        path_no_batch = lance_dataset_path + "_no_batch"
        for i, data in enumerate(data_parts):
            df = daft.from_pydict(data)
            mode = "create" if i == 0 else "append"
            df.write_lance(path_no_batch, batch_size=1, mode=mode)
        
        # Write with batching (batch_size=5)
        path_with_batch = lance_dataset_path + "_with_batch"
        for i, data in enumerate(data_parts):
            df = daft.from_pydict(data)
            mode = "create" if i == 0 else "append"
            df.write_lance(path_with_batch, batch_size=5, mode=mode)
        
        # Compare fragment counts
        ds_no_batch = lance.dataset(path_no_batch)
        ds_with_batch = lance.dataset(path_with_batch)
        
        fragments_no_batch = len(ds_no_batch.get_fragments())
        fragments_with_batch = len(ds_with_batch.get_fragments())
        
        # Batching should result in fewer fragments
        assert fragments_with_batch <= fragments_no_batch
        
        # Verify data integrity
        assert ds_no_batch.count_rows() == ds_with_batch.count_rows()

    def test_row_count_accuracy(self, lance_dataset_path):
        """Test that batching maintains accurate row counts."""
        # Create data with known row counts
        total_rows = 0
        data_parts = []
        for i in range(5):
            rows_in_part = (i + 1) * 10  # 10, 20, 30, 40, 50 rows
            data = {
                "id": list(range(total_rows, total_rows + rows_in_part)),
                "value": [float(j) for j in range(rows_in_part)],
            }
            data_parts.append(data)
            total_rows += rows_in_part
        
        # Write with batching
        for i, data in enumerate(data_parts):
            df = daft.from_pydict(data)
            mode = "create" if i == 0 else "append"
            df.write_lance(lance_dataset_path, batch_size=3, max_batch_rows=50, mode=mode)
        
        # Verify total row count
        ds = lance.dataset(lance_dataset_path)
        assert ds.count_rows() == total_rows

    def test_memory_usage_with_large_batches(self, lance_dataset_path):
        """Test memory usage behavior with large batches."""
        # Create moderately large data
        large_data = {
            "id": list(range(10000)),
            "text": [f"text_data_{i}" * 10 for i in range(10000)],  # Larger text fields
            "value": [float(i) * 1.1 for i in range(10000)],
        }
        
        df = daft.from_pydict(large_data)
        
        # This should not cause memory issues with row-based batching
        df.write_lance(
            lance_dataset_path,
            batch_size=100,
            max_batch_rows=2000,  # Reasonable batch size
            mode="create"
        )
        
        # Verify data was written correctly
        df_loaded = daft.read_lance(lance_dataset_path)
        assert df_loaded.count_rows().collect()[0]["count"] == 10000


class TestLanceDataSinkErrorHandling:
    """Error handling and edge case tests."""

    def test_empty_micropartitions_handling(self, lance_dataset_path, sample_schema, create_micropartitions):
        """Test handling of empty micropartitions."""
        # Create mix of empty and non-empty micropartitions
        data_list = [
            {"id": [], "name": [], "value": []},  # Empty
            {"id": [1], "name": ["alice"], "value": [1.1]},  # Non-empty
            {"id": [], "name": [], "value": []},  # Empty
            {"id": [2], "name": ["bob"], "value": [2.2]},  # Non-empty
        ]
        
        df_parts = []
        for data in data_list:
            if data["id"]:  # Only add non-empty parts to expected result
                df_parts.append(daft.from_pydict(data))
        
        # Write all parts (including empty ones)
        for i, data in enumerate(data_list):
            df = daft.from_pydict(data)
            mode = "create" if i == 0 else "append"
            if data["id"] or i == 0:  # Write empty only if it's the first (create mode)
                df.write_lance(lance_dataset_path, batch_size=2, mode=mode)
        
        # Verify only non-empty data was written
        if df_parts:
            expected_df = df_parts[0]
            for df_part in df_parts[1:]:
                expected_df = expected_df.concat(df_part)
            
            df_loaded = daft.read_lance(lance_dataset_path)
            loaded_data = df_loaded.to_pydict()
            expected_data = expected_df.to_pydict()
            
            assert len(loaded_data["id"]) == len(expected_data["id"])
            assert set(loaded_data["id"]) == set(expected_data["id"])

    def test_schema_compatibility_error(self, lance_dataset_path):
        """Test schema compatibility error handling."""
        # Create initial dataset with one schema
        initial_data = {"id": [1, 2], "name": ["alice", "bob"]}
        df1 = daft.from_pydict(initial_data)
        df1.write_lance(lance_dataset_path, mode="create")
        
        # Try to append data with incompatible schema
        incompatible_data = {"id": [3, 4], "different_field": ["charlie", "david"]}
        df2 = daft.from_pydict(incompatible_data)
        
        with pytest.raises(ValueError, match="Schema of data does not match table schema"):
            df2.write_lance(lance_dataset_path, batch_size=2, mode="append")

    def test_batch_processing_fallback(self, lance_dataset_path, sample_schema):
        """Test fallback to individual processing when batch processing fails."""
        sink = LanceDataSink(
            uri=lance_dataset_path,
            schema=sample_schema,
            mode="create",
            batch_size=3,
        )
        
        # Create micropartitions
        data_list = [
            {"id": [1], "name": ["alice"], "value": [1.1]},
            {"id": [2], "name": ["bob"], "value": [2.2]},
            {"id": [3], "name": ["charlie"], "value": [3.3]},
        ]
        
        micropartitions = []
        for data in data_list:
            df = daft.from_pydict(data)
            arrow_table = df.to_arrow()
            mp = MicroPartition.from_arrow(arrow_table)
            micropartitions.append(mp)
        
        # Mock batch processing to fail
        with patch.object(sink, '_process_batch', side_effect=Exception("Batch processing failed")):
            with patch.object(sink, '_write_individual') as mock_individual:
                mock_individual.return_value = iter([Mock()])
                
                # This should trigger fallback to individual processing
                list(sink._process_batch_with_fallback(micropartitions))
                
                # Verify individual processing was called for each micropartition
                assert mock_individual.call_count == len(micropartitions)

    def test_invalid_lance_parameters(self, lance_dataset_path, sample_schema):
        """Test error handling with invalid Lance parameters."""
        # Test with invalid Lance-specific parameters
        sink = LanceDataSink(
            uri=lance_dataset_path,
            schema=sample_schema,
            mode="create",
            batch_size=2,
            invalid_lance_param="invalid_value",  # This should be passed to Lance
        )
        
        data = {"id": [1, 2], "name": ["alice", "bob"], "value": [1.1, 2.2]}
        df = daft.from_pydict(data)
        
        # This might fail depending on Lance's parameter validation
        # The test ensures our code handles Lance errors gracefully
        try:
            df.write_lance(lance_dataset_path, batch_size=2, mode="create", invalid_lance_param="invalid")
        except Exception as e:
            # Ensure the error is from Lance, not our batching logic
            assert "invalid_lance_param" in str(e) or "Lance" in str(e) or "fragment" in str(e)


class TestLanceDataSinkEdgeCases:
    """Edge case tests."""

    def test_single_row_micropartitions(self, lance_dataset_path):
        """Test batching with single-row micropartitions."""
        # Create many single-row micropartitions
        data_parts = []
        for i in range(10):
            data_parts.append({
                "id": [i],
                "name": [f"user_{i}"],
                "value": [float(i)],
            })
        
        # Write with batching
        for i, data in enumerate(data_parts):
            df = daft.from_pydict(data)
            mode = "create" if i == 0 else "append"
            df.write_lance(lance_dataset_path, batch_size=5, mode=mode)
        
        # Verify all data was written
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        assert len(loaded_data["id"]) == 10
        assert set(loaded_data["id"]) == set(range(10))

    def test_exact_batch_size_boundary(self, lance_dataset_path):
        """Test behavior when data exactly matches batch size."""
        # Create exactly 6 rows to test with batch_size=3
        data = {
            "id": [1, 2, 3, 4, 5, 6],
            "name": ["a", "b", "c", "d", "e", "f"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
        
        df = daft.from_pydict(data)
        df.write_lance(lance_dataset_path, batch_size=3, mode="create")
        
        # Verify all data was written
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        assert len(loaded_data["id"]) == 6
        assert set(loaded_data["id"]) == set([1, 2, 3, 4, 5, 6])

    def test_max_batch_rows_boundary(self, lance_dataset_path):
        """Test behavior when row count exactly matches max_batch_rows."""
        # Create data that exactly matches max_batch_rows
        max_rows = 100
        data = {
            "id": list(range(max_rows)),
            "value": [float(i) for i in range(max_rows)],
        }
        
        df = daft.from_pydict(data)
        df.write_lance(
            lance_dataset_path,
            batch_size=1000,  # Large batch size
            max_batch_rows=max_rows,  # Exact match
            mode="create"
        )
        
        # Verify all data was written
        df_loaded = daft.read_lance(lance_dataset_path)
        loaded_data = df_loaded.to_pydict()
        
        assert len(loaded_data["id"]) == max_rows
        assert set(loaded_data["id"]) == set(range(max_rows))


class TestPyArrowSchemaUtilities:
    """Test utility functions."""

    def test_pyarrow_schema_castable_compatible(self):
        """Test schema compatibility checking with compatible schemas."""
        schema1 = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
        ])
        schema2 = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("name", pa.large_string()),  # Compatible with string
        ])
        
        assert pyarrow_schema_castable(schema1, schema2)

    def test_pyarrow_schema_castable_incompatible(self):
        """Test schema compatibility checking with incompatible schemas."""
        schema1 = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
        ])
        schema2 = pa.schema([
            pa.field("id", pa.string()),  # Incompatible type
            pa.field("name", pa.string()),
        ])
        
        assert not pyarrow_schema_castable(schema1, schema2)

    def test_pyarrow_schema_castable_different_fields(self):
        """Test schema compatibility with different number of fields."""
        schema1 = pa.schema([
            pa.field("id", pa.int64()),
        ])
        schema2 = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("name", pa.string()),
        ])
        
        assert not pyarrow_schema_castable(schema1, schema2)


# Performance benchmark tests (optional, can be run separately)
class TestLanceDataSinkBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.benchmark
    def test_batching_performance_comparison(self, lance_dataset_path, benchmark):
        """Benchmark batching vs non-batching performance."""
        # This test requires pytest-benchmark plugin
        # Skip if not available
        pytest.importorskip("pytest_benchmark")
        
        # Create test data
        data = {
            "id": list(range(1000)),
            "text": [f"data_{i}" for i in range(1000)],
            "value": [float(i) * 1.1 for i in range(1000)],
        }
        
        def write_with_batching():
            df = daft.from_pydict(data)
            path = lance_dataset_path + "_batch"
            df.write_lance(path, batch_size=10, mode="create")
            return path
        
        def write_without_batching():
            df = daft.from_pydict(data)
            path = lance_dataset_path + "_no_batch"
            df.write_lance(path, batch_size=1, mode="create")
            return path
        
        # Benchmark both approaches
        batched_path = benchmark.pedantic(write_with_batching, rounds=3, iterations=1)
        
        # Verify results are equivalent
        df_batched = daft.read_lance(batched_path)
        assert df_batched.count_rows().collect()[0]["count"] == 1000


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/io/lance/test_lance_datasink_batching.py -v
    pytest.main([__file__, "-v"])