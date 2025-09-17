"""
Test configuration and fixtures for Lance DataSink batching tests.

This module provides shared fixtures and configuration for all Lance DataSink
batching tests, ensuring consistent test setup and teardown.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pytest

import daft
from daft.datatype import DataType
from daft.schema import Schema

# Skip tests if Lance is not available
try:
    import lance
    LANCE_AVAILABLE = True
except ImportError:
    LANCE_AVAILABLE = False

PYARROW_LOWER_BOUND_SKIP = tuple(int(s) for s in pa.__version__.split(".") if s.isnumeric()) < (9, 0, 0)

# Global skip condition for all Lance tests
pytestmark = pytest.mark.skipif(
    PYARROW_LOWER_BOUND_SKIP or not LANCE_AVAILABLE, 
    reason="lance not supported on old versions of pyarrow or lance not installed"
)


@pytest.fixture(scope="function")
def lance_temp_dir(tmp_path_factory):
    """Create a temporary directory for Lance dataset tests."""
    tmp_dir = tmp_path_factory.mktemp("lance_tests")
    yield str(tmp_dir)


@pytest.fixture(scope="function")
def sample_schemas():
    """Provide various sample schemas for testing."""
    return {
        "simple": Schema._from_field_name_and_types([
            ("id", DataType.int64()),
            ("name", DataType.string()),
            ("value", DataType.float64()),
        ]),
        "complex": Schema._from_field_name_and_types([
            ("id", DataType.int64()),
            ("timestamp", DataType.string()),
            ("category", DataType.string()),
            ("value", DataType.float64()),
            ("flag", DataType.bool()),
            ("tags", DataType.list(DataType.string())),
        ]),
        "nullable": Schema._from_field_name_and_types([
            ("id", DataType.int64()),
            ("optional_text", DataType.string()),
            ("optional_number", DataType.float64()),
            ("required_field", DataType.string()),
        ]),
    }


@pytest.fixture(scope="function")
def sample_data_sets():
    """Provide various sample datasets for testing."""
    return {
        "tiny": {
            "id": [1, 2, 3],
            "name": ["alice", "bob", "charlie"],
            "value": [1.1, 2.2, 3.3],
        },
        "small": {
            "id": list(range(50)),
            "name": [f"user_{i}" for i in range(50)],
            "value": [float(i) * 1.1 for i in range(50)],
        },
        "medium": {
            "id": list(range(500)),
            "name": [f"user_{i}" for i in range(500)],
            "value": [float(i) * 1.1 for i in range(500)],
        },
        "with_nulls": {
            "id": [1, 2, 3, 4, 5],
            "optional_text": ["a", None, "c", None, "e"],
            "optional_number": [1.0, None, 3.0, None, 5.0],
            "required_field": ["x", "y", "z", "w", "v"],
        },
        "complex_types": {
            "id": [1, 2, 3, 4, 5],
            "timestamp": ["2024-01-01T00:00:00", "2024-01-01T01:00:00", 
                         "2024-01-01T02:00:00", "2024-01-01T03:00:00", 
                         "2024-01-01T04:00:00"],
            "category": ["A", "B", "A", "C", "B"],
            "value": [1.1, 2.2, 3.3, 4.4, 5.5],
            "flag": [True, False, True, False, True],
            "tags": [["tag1", "tag2"], ["tag3"], [], ["tag4", "tag5", "tag6"], ["tag7"]],
        },
    }


@pytest.fixture
def create_test_dataframes(sample_data_sets):
    """Factory function to create test DataFrames."""
    def _create_dataframes(data_key: str, n_parts: int = 1):
        """Create n_parts DataFrames from the specified dataset."""
        if data_key not in sample_data_sets:
            raise ValueError(f"Unknown data key: {data_key}")
        
        data = sample_data_sets[data_key]
        total_rows = len(data[list(data.keys())[0]])
        
        if n_parts == 1:
            return [daft.from_pydict(data)]
        
        # Split data into n_parts
        rows_per_part = total_rows // n_parts
        dataframes = []
        
        for i in range(n_parts):
            start_idx = i * rows_per_part
            if i == n_parts - 1:  # Last part gets remaining rows
                end_idx = total_rows
            else:
                end_idx = start_idx + rows_per_part
            
            part_data = {}
            for key, values in data.items():
                part_data[key] = values[start_idx:end_idx]
            
            if part_data[list(part_data.keys())[0]]:  # Only add non-empty parts
                dataframes.append(daft.from_pydict(part_data))
        
        return dataframes
    
    return _create_dataframes


@pytest.fixture
def batch_size_scenarios():
    """Provide various batch size scenarios for testing."""
    return {
        "no_batching": {"batch_size": 1, "max_batch_rows": 100000},
        "small_batches": {"batch_size": 3, "max_batch_rows": 100000},
        "medium_batches": {"batch_size": 10, "max_batch_rows": 100000},
        "large_batches": {"batch_size": 50, "max_batch_rows": 100000},
        "row_limited": {"batch_size": 100, "max_batch_rows": 25},
        "very_row_limited": {"batch_size": 100, "max_batch_rows": 5},
    }


@pytest.fixture
def performance_test_data():
    """Generate performance test data with known characteristics."""
    def _generate_data(n_rows: int, text_size: int = 1, complexity: str = "simple"):
        """Generate test data with specified characteristics."""
        if complexity == "simple":
            return {
                "id": list(range(n_rows)),
                "value": [float(i) * 1.1 for i in range(n_rows)],
                "text": [f"text_{i}" * text_size for i in range(n_rows)],
            }
        elif complexity == "medium":
            return {
                "id": list(range(n_rows)),
                "category": [f"cat_{i % 10}" for i in range(n_rows)],
                "value": [float(i) * 1.1 for i in range(n_rows)],
                "text": [f"text_data_{i}" * text_size for i in range(n_rows)],
                "flag": [i % 2 == 0 for i in range(n_rows)],
            }
        elif complexity == "complex":
            return {
                "id": list(range(n_rows)),
                "timestamp": [f"2024-01-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00" for i in range(n_rows)],
                "category": [f"category_{i % 20}" for i in range(n_rows)],
                "subcategory": [f"subcat_{i % 5}" for i in range(n_rows)],
                "value": [float(i) * 1.1 for i in range(n_rows)],
                "secondary_value": [float(i) * 2.3 for i in range(n_rows)],
                "text_field": [f"complex_text_data_{i}" * text_size for i in range(n_rows)],
                "flag_a": [i % 2 == 0 for i in range(n_rows)],
                "flag_b": [i % 3 == 0 for i in range(n_rows)],
                "tags": [[f"tag_{i % 5}", f"tag_{(i + 1) % 5}"] for i in range(n_rows)],
            }
        else:
            raise ValueError(f"Unknown complexity: {complexity}")
    
    return _generate_data


@pytest.fixture
def verify_lance_dataset():
    """Utility function to verify Lance dataset integrity."""
    def _verify_dataset(dataset_path: str, expected_row_count: int = None, 
                       expected_columns: list = None, check_fragments: bool = True):
        """Verify Lance dataset integrity and characteristics."""
        if not LANCE_AVAILABLE:
            pytest.skip("Lance not available")
        
        # Check that dataset exists and is readable
        ds = lance.dataset(dataset_path)
        
        # Verify row count
        actual_rows = ds.count_rows()
        if expected_row_count is not None:
            assert actual_rows == expected_row_count, f"Expected {expected_row_count} rows, got {actual_rows}"
        
        # Verify columns
        if expected_columns is not None:
            actual_columns = set(ds.schema.names)
            expected_columns_set = set(expected_columns)
            assert actual_columns == expected_columns_set, f"Column mismatch. Expected {expected_columns_set}, got {actual_columns}"
        
        # Check fragments
        if check_fragments:
            fragments = ds.get_fragments()
            assert len(fragments) > 0, "Dataset should have at least one fragment"
        
        return {
            "row_count": actual_rows,
            "fragment_count": len(ds.get_fragments()) if check_fragments else None,
            "schema": ds.schema,
            "version": ds.version,
        }
    
    return _verify_dataset


@pytest.fixture
def compare_datasets():
    """Utility function to compare two Lance datasets."""
    def _compare_datasets(path1: str, path2: str, check_fragment_count: bool = False):
        """Compare two Lance datasets for data integrity."""
        if not LANCE_AVAILABLE:
            pytest.skip("Lance not available")
        
        ds1 = lance.dataset(path1)
        ds2 = lance.dataset(path2)
        
        # Compare row counts
        rows1 = ds1.count_rows()
        rows2 = ds2.count_rows()
        assert rows1 == rows2, f"Row count mismatch: {rows1} vs {rows2}"
        
        # Compare schemas
        schema1 = ds1.schema
        schema2 = ds2.schema
        assert schema1.equals(schema2), f"Schema mismatch: {schema1} vs {schema2}"
        
        # Optionally compare fragment counts
        if check_fragment_count:
            frags1 = len(ds1.get_fragments())
            frags2 = len(ds2.get_fragments())
            return {
                "row_count": rows1,
                "fragment_count_1": frags1,
                "fragment_count_2": frags2,
                "fragment_ratio": frags2 / frags1 if frags1 > 0 else 1.0,
            }
        
        return {"row_count": rows1}
    
    return _compare_datasets


@pytest.fixture(scope="session")
def lance_test_environment():
    """Set up test environment for Lance tests."""
    if not LANCE_AVAILABLE:
        pytest.skip("Lance not available for testing")
    
    # Set up any global test configuration
    original_env = os.environ.copy()
    
    # Set test-specific environment variables if needed
    test_env = {
        "DAFT_RUNNER": os.getenv("DAFT_RUNNER", "native"),
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Pytest markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "lance_batching: mark test as Lance DataSink batching test"
    )
    config.addinivalue_line(
        "markers", "lance_performance: mark test as Lance performance test"
    )
    config.addinivalue_line(
        "markers", "lance_integration: mark test as Lance integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Parametrized fixtures for comprehensive testing
@pytest.fixture(params=[1, 5, 10, 25])
def batch_size_param(request):
    """Parametrized batch size for testing."""
    return request.param


@pytest.fixture(params=[100, 500, 1000, 5000])
def max_batch_rows_param(request):
    """Parametrized max batch rows for testing."""
    return request.param


@pytest.fixture(params=["create", "append", "overwrite"])
def write_mode_param(request):
    """Parametrized write mode for testing."""
    return request.param