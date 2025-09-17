"""
Performance-focused tests for Lance DataSink batching functionality.

This module contains tests specifically designed to validate the performance
improvements and characteristics of the batching mechanism.
"""

from __future__ import annotations

import time
from pathlib import Path

import pyarrow as pa
import pytest

import daft

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
    tmp_dir = tmp_path_factory.mktemp("lance_performance")
    yield str(tmp_dir)


class TestLanceDataSinkFragmentReduction:
    """Test that batching reduces the number of Lance fragments."""

    def test_fragment_count_with_different_batch_sizes(self, lance_dataset_path):
        """Test fragment count reduction with various batch sizes."""
        # Create test data that would create many small fragments without batching
        n_parts = 50
        data_parts = []
        for i in range(n_parts):
            data_parts.append({
                "part_id": [i],
                "timestamp": [f"2024-01-01T{i:02d}:00:00"],
                "value": [float(i) * 1.1],
                "category": [f"cat_{i % 5}"],
            })
        
        # Test different batch sizes
        batch_sizes = [1, 5, 10, 25]
        fragment_counts = {}
        
        for batch_size in batch_sizes:
            path = f"{lance_dataset_path}_batch_{batch_size}"
            
            # Write data with current batch size
            for i, data in enumerate(data_parts):
                df = daft.from_pydict(data)
                mode = "create" if i == 0 else "append"
                df.write_lance(path, batch_size=batch_size, mode=mode)
            
            # Count fragments
            ds = lance.dataset(path)
            fragment_counts[batch_size] = len(ds.get_fragments())
            
            # Verify data integrity
            assert ds.count_rows() == n_parts
        
        # Verify that larger batch sizes result in fewer fragments
        assert fragment_counts[1] >= fragment_counts[5]
        assert fragment_counts[5] >= fragment_counts[10]
        assert fragment_counts[10] >= fragment_counts[25]
        
        # With batch_size=1, should have close to n_parts fragments
        # With larger batch sizes, should have significantly fewer
        assert fragment_counts[25] < fragment_counts[1] * 0.5  # At least 50% reduction

    def test_row_based_batching_fragment_reduction(self, lance_dataset_path):
        """Test fragment reduction with row-based batching."""
        # Create data with varying row counts per part
        data_parts = []
        total_rows = 0
        
        part_row_counts = [10, 25, 15, 30, 20, 35, 5, 40]  # Varying sizes
        for i, row_count in enumerate(part_row_counts):
            data = {
                "part_id": [i] * row_count,
                "row_in_part": list(range(row_count)),
                "global_row": list(range(total_rows, total_rows + row_count)),
                "value": [float(j + i * 100) for j in range(row_count)],
            }
            data_parts.append(data)
            total_rows += row_count
        
        # Test with row-based batching
        path_row_batch = f"{lance_dataset_path}_row_batch"
        for i, data in enumerate(data_parts):
            df = daft.from_pydict(data)
            mode = "create" if i == 0 else "append"
            df.write_lance(
                path_row_batch,
                batch_size=100,  # Large batch size
                max_batch_rows=50,  # Row-based limit
                mode=mode
            )
        
        # Test without batching
        path_no_batch = f"{lance_dataset_path}_no_batch"
        for i, data in enumerate(data_parts):
            df = daft.from_pydict(data)
            mode = "create" if i == 0 else "append"
            df.write_lance(path_no_batch, batch_size=1, mode=mode)
        
        # Compare fragment counts
        ds_row_batch = lance.dataset(path_row_batch)
        ds_no_batch = lance.dataset(path_no_batch)
        
        fragments_row_batch = len(ds_row_batch.get_fragments())
        fragments_no_batch = len(ds_no_batch.get_fragments())
        
        # Row-based batching should reduce fragments
        assert fragments_row_batch < fragments_no_batch
        
        # Verify data integrity
        assert ds_row_batch.count_rows() == total_rows
        assert ds_no_batch.count_rows() == total_rows

    def test_optimal_batch_size_analysis(self, lance_dataset_path):
        """Analyze optimal batch size for different data characteristics."""
        # Create datasets with different characteristics
        scenarios = {
            "small_rows": {"row_size": 10, "n_parts": 100},
            "medium_rows": {"row_size": 50, "n_parts": 40},
            "large_rows": {"row_size": 200, "n_parts": 20},
        }
        
        results = {}
        
        for scenario_name, params in scenarios.items():
            row_size = params["row_size"]
            n_parts = params["n_parts"]
            
            # Create data parts
            data_parts = []
            for i in range(n_parts):
                data = {
                    "part_id": [i] * row_size,
                    "row_id": list(range(i * row_size, (i + 1) * row_size)),
                    "data": [f"data_{j}_{scenario_name}" for j in range(row_size)],
                    "value": [float(j + i * 1000) for j in range(row_size)],
                }
                data_parts.append(data)
            
            # Test different batch sizes
            batch_sizes = [1, 5, 10, 20]
            scenario_results = {}
            
            for batch_size in batch_sizes:
                path = f"{lance_dataset_path}_{scenario_name}_batch_{batch_size}"
                
                start_time = time.time()
                
                for i, data in enumerate(data_parts):
                    df = daft.from_pydict(data)
                    mode = "create" if i == 0 else "append"
                    df.write_lance(path, batch_size=batch_size, mode=mode)
                
                write_time = time.time() - start_time
                
                # Analyze results
                ds = lance.dataset(path)
                fragment_count = len(ds.get_fragments())
                row_count = ds.count_rows()
                
                scenario_results[batch_size] = {
                    "write_time": write_time,
                    "fragment_count": fragment_count,
                    "row_count": row_count,
                }
            
            results[scenario_name] = scenario_results
        
        # Verify that batching improves performance metrics
        for scenario_name, scenario_results in results.items():
            # Fragment count should decrease with larger batch sizes
            fragments_1 = scenario_results[1]["fragment_count"]
            fragments_20 = scenario_results[20]["fragment_count"]
            assert fragments_20 <= fragments_1
            
            # All scenarios should have correct row counts
            expected_rows = scenarios[scenario_name]["row_size"] * scenarios[scenario_name]["n_parts"]
            for batch_size, metrics in scenario_results.items():
                assert metrics["row_count"] == expected_rows


class TestLanceDataSinkWritePerformance:
    """Test write performance characteristics of batching."""

    def test_write_time_comparison(self, lance_dataset_path):
        """Compare write times between batched and non-batched approaches."""
        # Create consistent test data
        n_parts = 30
        rows_per_part = 50
        
        data_parts = []
        for i in range(n_parts):
            data = {
                "id": list(range(i * rows_per_part, (i + 1) * rows_per_part)),
                "timestamp": [f"2024-01-01T{(i * rows_per_part + j) % 24:02d}:00:00" for j in range(rows_per_part)],
                "category": [f"category_{(i + j) % 10}" for j in range(rows_per_part)],
                "value": [float(i * rows_per_part + j) * 1.1 for j in range(rows_per_part)],
                "text_data": [f"text_data_{i}_{j}" * 5 for j in range(rows_per_part)],
            }
            data_parts.append(data)
        
        # Test non-batched approach
        path_no_batch = f"{lance_dataset_path}_timing_no_batch"
        start_time = time.time()
        
        for i, data in enumerate(data_parts):
            df = daft.from_pydict(data)
            mode = "create" if i == 0 else "append"
            df.write_lance(path_no_batch, batch_size=1, mode=mode)
        
        time_no_batch = time.time() - start_time
        
        # Test batched approach
        path_batch = f"{lance_dataset_path}_timing_batch"
        start_time = time.time()
        
        for i, data in enumerate(data_parts):
            df = daft.from_pydict(data)
            mode = "create" if i == 0 else "append"
            df.write_lance(path_batch, batch_size=10, max_batch_rows=200, mode=mode)
        
        time_batch = time.time() - start_time
        
        # Verify data integrity
        ds_no_batch = lance.dataset(path_no_batch)
        ds_batch = lance.dataset(path_batch)
        
        total_expected_rows = n_parts * rows_per_part
        assert ds_no_batch.count_rows() == total_expected_rows
        assert ds_batch.count_rows() == total_expected_rows
        
        # Performance analysis (batching should generally be faster or similar)
        # Note: In some cases, batching might be slightly slower due to overhead,
        # but it should provide better fragment organization
        fragments_no_batch = len(ds_no_batch.get_fragments())
        fragments_batch = len(ds_batch.get_fragments())
        
        # Batching should result in fewer fragments
        assert fragments_batch <= fragments_no_batch
        
        # Log performance metrics for analysis
        print(f"\nPerformance Comparison:")
        print(f"No Batching: {time_no_batch:.3f}s, {fragments_no_batch} fragments")
        print(f"With Batching: {time_batch:.3f}s, {fragments_batch} fragments")
        print(f"Fragment reduction: {((fragments_no_batch - fragments_batch) / fragments_no_batch * 100):.1f}%")

    def test_memory_usage_patterns(self, lance_dataset_path):
        """Test memory usage patterns with different batching configurations."""
        # Create data with varying memory footprints
        memory_scenarios = [
            {"name": "small_memory", "text_multiplier": 1, "rows": 1000},
            {"name": "medium_memory", "text_multiplier": 10, "rows": 500},
            {"name": "large_memory", "text_multiplier": 50, "rows": 200},
        ]
        
        for scenario in memory_scenarios:
            name = scenario["name"]
            text_mult = scenario["text_multiplier"]
            n_rows = scenario["rows"]
            
            # Create memory-intensive data
            data = {
                "id": list(range(n_rows)),
                "large_text": [f"memory_test_data_{'x' * text_mult}_{i}" for i in range(n_rows)],
                "category": [f"cat_{i % 5}" for i in range(n_rows)],
                "value": [float(i) * 1.1 for i in range(n_rows)],
            }
            
            df = daft.from_pydict(data)
            
            # Test with row-based memory management
            path = f"{lance_dataset_path}_{name}"
            
            # Use conservative row limits for memory management
            max_batch_rows = max(10, n_rows // 20)  # Limit batch size based on data size
            
            df.write_lance(
                path,
                batch_size=100,  # Large batch size
                max_batch_rows=max_batch_rows,  # Memory-conscious row limit
                mode="create"
            )
            
            # Verify data was written successfully
            ds = lance.dataset(path)
            assert ds.count_rows() == n_rows
            
            # Check that fragments are reasonable
            fragments = ds.get_fragments()
            assert len(fragments) > 0
            assert len(fragments) <= n_rows  # Sanity check

    def test_concurrent_write_performance(self, lance_dataset_path):
        """Test performance characteristics that simulate concurrent writes."""
        # Simulate multiple data streams being written
        n_streams = 5
        batches_per_stream = 10
        rows_per_batch = 20
        
        all_data = []
        
        # Generate interleaved data from multiple streams
        for batch_idx in range(batches_per_stream):
            for stream_idx in range(n_streams):
                data = {
                    "stream_id": [stream_idx] * rows_per_batch,
                    "batch_id": [batch_idx] * rows_per_batch,
                    "row_id": list(range(batch_idx * rows_per_batch, (batch_idx + 1) * rows_per_batch)),
                    "timestamp": [f"2024-01-01T{(stream_idx * 5 + batch_idx) % 24:02d}:00:00"] * rows_per_batch,
                    "data": [f"stream_{stream_idx}_batch_{batch_idx}_row_{i}" for i in range(rows_per_batch)],
                    "value": [float(stream_idx * 1000 + batch_idx * 100 + i) for i in range(rows_per_batch)],
                }
                all_data.append(data)
        
        # Test with batching optimized for this pattern
        start_time = time.time()
        
        for i, data in enumerate(all_data):
            df = daft.from_pydict(data)
            mode = "create" if i == 0 else "append"
            df.write_lance(
                lance_dataset_path,
                batch_size=n_streams,  # Batch size matches stream count
                max_batch_rows=rows_per_batch * n_streams,  # Allow full batch accumulation
                mode=mode
            )
        
        write_time = time.time() - start_time
        
        # Verify results
        ds = lance.dataset(lance_dataset_path)
        total_expected_rows = n_streams * batches_per_stream * rows_per_batch
        assert ds.count_rows() == total_expected_rows
        
        # Check fragment organization
        fragments = ds.get_fragments()
        
        # Should have significantly fewer fragments than total writes
        total_writes = len(all_data)
        assert len(fragments) < total_writes
        
        print(f"\nConcurrent Write Simulation:")
        print(f"Total writes: {total_writes}")
        print(f"Fragments created: {len(fragments)}")
        print(f"Write time: {write_time:.3f}s")
        print(f"Fragment efficiency: {(total_writes - len(fragments)) / total_writes * 100:.1f}% reduction")


class TestLanceDataSinkScalability:
    """Test scalability characteristics of the batching mechanism."""

    def test_large_dataset_scalability(self, lance_dataset_path):
        """Test batching behavior with larger datasets."""
        # Create a moderately large dataset
        n_rows = 5000
        batch_size = 100
        
        # Generate data in chunks to simulate real-world usage
        chunk_size = 500
        n_chunks = n_rows // chunk_size
        
        total_write_time = 0
        
        for chunk_idx in range(n_chunks):
            start_row = chunk_idx * chunk_size
            end_row = min(start_row + chunk_size, n_rows)
            chunk_rows = end_row - start_row
            
            data = {
                "id": list(range(start_row, end_row)),
                "chunk_id": [chunk_idx] * chunk_rows,
                "timestamp": [f"2024-01-{(start_row + i) % 28 + 1:02d}T00:00:00" for i in range(chunk_rows)],
                "category": [f"category_{(start_row + i) % 20}" for i in range(chunk_rows)],
                "value": [float(start_row + i) * 1.1 for i in range(chunk_rows)],
                "text_field": [f"text_data_{start_row + i}" * 3 for i in range(chunk_rows)],
            }
            
            df = daft.from_pydict(data)
            
            start_time = time.time()
            mode = "create" if chunk_idx == 0 else "append"
            df.write_lance(
                lance_dataset_path,
                batch_size=batch_size,
                max_batch_rows=200,
                mode=mode
            )
            chunk_time = time.time() - start_time
            total_write_time += chunk_time
        
        # Verify final dataset
        ds = lance.dataset(lance_dataset_path)
        assert ds.count_rows() == n_rows
        
        fragments = ds.get_fragments()
        
        # Performance metrics
        avg_rows_per_fragment = n_rows / len(fragments) if fragments else 0
        
        print(f"\nLarge Dataset Scalability Test:")
        print(f"Total rows: {n_rows}")
        print(f"Total fragments: {len(fragments)}")
        print(f"Average rows per fragment: {avg_rows_per_fragment:.1f}")
        print(f"Total write time: {total_write_time:.3f}s")
        print(f"Rows per second: {n_rows / total_write_time:.1f}")
        
        # Verify reasonable fragment organization
        assert len(fragments) < n_chunks  # Should have fewer fragments than chunks
        assert avg_rows_per_fragment > batch_size  # Should batch effectively

    def test_varying_data_sizes_scalability(self, lance_dataset_path):
        """Test scalability with varying data sizes within the same dataset."""
        # Create data with exponentially increasing sizes
        data_parts = []
        total_rows = 0
        
        for i in range(8):  # 8 parts with increasing sizes
            part_rows = 10 * (2 ** i)  # 10, 20, 40, 80, 160, 320, 640, 1280
            data = {
                "part_id": [i] * part_rows,
                "row_in_part": list(range(part_rows)),
                "global_row": list(range(total_rows, total_rows + part_rows)),
                "size_category": [f"size_{i}"] * part_rows,
                "value": [float(total_rows + j) * 1.1 for j in range(part_rows)],
                "text_data": [f"part_{i}_row_{j}_{'x' * (i + 1)}" for j in range(part_rows)],
            }
            data_parts.append(data)
            total_rows += part_rows
        
        # Write with adaptive batching
        for i, data in enumerate(data_parts):
            df = daft.from_pydict(data)
            mode = "create" if i == 0 else "append"
            
            # Adjust batch parameters based on data size
            part_rows = len(data["part_id"])
            adaptive_batch_size = min(10, max(1, part_rows // 50))
            adaptive_max_rows = min(500, part_rows)
            
            df.write_lance(
                lance_dataset_path,
                batch_size=adaptive_batch_size,
                max_batch_rows=adaptive_max_rows,
                mode=mode
            )
        
        # Verify scalability
        ds = lance.dataset(lance_dataset_path)
        assert ds.count_rows() == total_rows
        
        fragments = ds.get_fragments()
        
        print(f"\nVarying Data Sizes Scalability Test:")
        print(f"Total rows: {total_rows}")
        print(f"Data parts: {len(data_parts)}")
        print(f"Fragments: {len(fragments)}")
        print(f"Fragment efficiency: {len(data_parts) / len(fragments):.2f}x consolidation")
        
        # Should have consolidated multiple parts into fewer fragments
        assert len(fragments) <= len(data_parts)


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/io/lance/test_lance_datasink_performance.py -v -s
    pytest.main([__file__, "-v", "-s"])