
from __future__ import annotations

import logging
import pathlib
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal

import lance

from daft.context import get_context
from daft.datatype import DataType
from daft.dependencies import pa
from daft.io import DataSink
from daft.io.sink import WriteResult
from daft.recordbatch import MicroPartition
from daft.schema import Schema

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

    from daft.daft import IOConfig

logger = logging.getLogger(__name__)


def pyarrow_schema_castable(src: pa.Schema, dst: pa.Schema) -> bool:
    """Check if source schema can be cast to destination schema."""
    if len(src) != len(dst):
        return False
    for src_field, dst_field in zip(src, dst):
        empty_array = pa.array([], type=src_field.type)
        try:
            empty_array.cast(dst_field.type)
        except Exception:
            return False
    return True


class LanceDataSink(DataSink[list[lance.FragmentMetadata]]):
    """WriteSink for writing data to a Lance dataset."""

    def _import_lance(self) -> ModuleType:
        try:
            import lance

            return lance
        except ImportError:
            raise ImportError("lance is not installed. Please install lance using `pip install daft[lance]`")

    def __init__(
        self,
        uri: str | pathlib.Path,
        schema: Schema,
        mode: Literal["create", "append", "overwrite"],
        io_config: IOConfig | None = None,
        batch_size: int = 1,
        max_batch_rows: int = 100000,
        **kwargs: Any,
    ) -> None:
        from daft.io.object_store_options import io_config_to_storage_options

        lance = self._import_lance()
        if not isinstance(uri, (str, pathlib.Path)):
            raise TypeError(f"Expected URI to be str or pathlib.Path, got {type(uri)}")
        
        # Validate batch_size parameter
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
        
        # Validate max_batch_rows parameter
        if not isinstance(max_batch_rows, int) or max_batch_rows < 1:
            raise ValueError(f"max_batch_rows must be a positive integer, got {max_batch_rows}")
        
        self._table_uri = str(uri)
        self._mode = mode
        self._io_config = get_context().daft_planning_config.default_io_config if io_config is None else io_config
        self._kwargs = kwargs
        self._batch_size = batch_size
        self._max_batch_rows = max_batch_rows

        self._storage_options = io_config_to_storage_options(self._io_config, self._table_uri)

        self._pyarrow_schema = schema.to_pyarrow_schema()

        try:
            table = lance.dataset(self._table_uri, storage_options=self._storage_options)
        except ValueError:
            table = None

        self._version = 0
        self._table_schema: pa.Schema | None = None
        if table is not None:
            self._table_schema = table.schema
            self._version = table.latest_version
            if not pyarrow_schema_castable(self._pyarrow_schema, self._table_schema) and not (
                self._mode == "overwrite"
            ):
                raise ValueError(
                    "Schema of data does not match table schema\n"
                    f"Data schema:\n{self._pyarrow_schema}\nTable Schema:\n{self._table_schema}"
                )

        self._schema = Schema._from_field_name_and_types(
            [
                ("num_fragments", DataType.int64()),
                ("num_deleted_rows", DataType.int64()),
                ("num_small_files", DataType.int64()),
                ("version", DataType.int64()),
            ]
        )

    def name(self) -> str:
        """Optional custom sink name."""
        return f"Lance Write (batch_size={self._batch_size}, max_batch_rows={self._max_batch_rows})"

    def schema(self) -> Schema:
        return self._schema

    def write(self, micropartitions: Iterator[MicroPartition]) -> Iterator[WriteResult[list[lance.FragmentMetadata]]]:
        """Writes fragments from the given micropartitions with batching support.
        
        Args:
            micropartitions: Iterator of micropartitions to write
            
        Yields:
            WriteResult: Results from writing batches of micropartitions
        """
        if self._batch_size <= 1:
            # Use original logic for backward compatibility
            yield from self._write_individual(micropartitions)
        else:
            # Use new batching logic
            yield from self._write_batched(micropartitions)

    def _write_individual(
        self, micropartitions: Iterator[MicroPartition]
    ) -> Iterator[WriteResult[list[lance.FragmentMetadata]]]:
        """Original write logic for individual micropartitions (backward compatibility)."""
        lance = self._import_lance()

        for micropartition in micropartitions:
            # Skip empty micropartitions
            if micropartition.num_rows() == 0:
                logger.debug("Skipping empty micropartition")
                continue
                
            arrow_table = pa.Table.from_batches(
                micropartition.to_arrow().to_batches(),
                self._pyarrow_schema,
            )
            if self._table_schema is not None:
                arrow_table = arrow_table.cast(self._table_schema)

            bytes_written = arrow_table.nbytes
            rows_written = arrow_table.num_rows

            fragments = lance.fragment.write_fragments(
                arrow_table,
                dataset_uri=self._table_uri,
                mode=self._mode,
                storage_options=self._storage_options,
                **self._kwargs,
            )
            yield WriteResult(
                result=fragments,
                bytes_written=bytes_written,
                rows_written=rows_written,
            )

    def _write_batched(
        self, micropartitions: Iterator[MicroPartition]
    ) -> Iterator[WriteResult[list[lance.FragmentMetadata]]]:
        """New batched write logic that combines multiple micropartitions."""
        batch = []
        batch_row_count = 0
        
        for micropartition in micropartitions:
            # Skip empty micropartitions
            if micropartition.num_rows() == 0:
                logger.debug("Skipping empty micropartition in batch")
                continue
            
            # Get actual row count
            mp_rows = micropartition.num_rows()
            
            # Add to current batch
            batch.append(micropartition)
            batch_row_count += mp_rows
            
            # Check if we should flush the batch
            if self._should_flush_batch(batch, batch_row_count):
                yield from self._process_batch_with_fallback(batch)
                batch = []
                batch_row_count = 0
        
        # Process remaining partial batch
        if batch:
            yield from self._process_batch_with_fallback(batch)

    def _should_flush_batch(self, batch: list[MicroPartition], current_rows: int) -> bool:
        """Determine if the current batch should be flushed."""
        return (len(batch) >= self._batch_size or 
                current_rows >= self._max_batch_rows)

    def _process_batch_with_fallback(
        self, batch: list[MicroPartition]
    ) -> Iterator[WriteResult[list[lance.FragmentMetadata]]]:
        """Process a batch with fallback to individual processing on failure."""
        try:
            yield self._process_batch(batch)
        except Exception as e:
            logger.warning(f"Batch processing failed for {len(batch)} micropartitions, "
                         f"falling back to individual processing: {e}")
            # Fallback to individual processing
            for mp in batch:
                try:
                    yield from self._write_individual(iter([mp]))
                except Exception as individual_error:
                    logger.error(f"Failed to write individual micropartition: {individual_error}")
                    # Re-raise the individual error as it's more specific
                    raise individual_error from e

    def _process_batch(self, batch: list[MicroPartition]) -> WriteResult[list[lance.FragmentMetadata]]:
        """Process a batch of micropartitions by combining them into a single write."""
        if not batch:
            raise ValueError("Cannot process empty batch")
        
        lance = self._import_lance()
        
        # Convert all micropartitions to Arrow tables
        arrow_tables = []
        total_bytes = 0
        total_rows = 0
        
        for mp in batch:
            arrow_table = pa.Table.from_batches(
                mp.to_arrow().to_batches(),
                self._pyarrow_schema,
            )
            if self._table_schema is not None:
                arrow_table = arrow_table.cast(self._table_schema)
            
            arrow_tables.append(arrow_table)
            total_bytes += arrow_table.nbytes
            total_rows += arrow_table.num_rows
        
        # Combine Arrow tables
        if len(arrow_tables) == 1:
            combined_table = arrow_tables[0]
        else:
            try:
                combined_table = pa.concat_tables(arrow_tables)
            except Exception as e:
                raise RuntimeError(f"Failed to concatenate {len(arrow_tables)} Arrow tables: {e}") from e
        
        # Write combined table to Lance
        try:
            fragments = lance.fragment.write_fragments(
                combined_table,
                dataset_uri=self._table_uri,
                mode=self._mode,
                storage_options=self._storage_options,
                **self._kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to write combined table with {total_rows} rows "
                             f"and {total_bytes} bytes to Lance: {e}") from e
        
        logger.debug(f"Successfully wrote batch of {len(batch)} micropartitions "
                    f"({total_rows} rows, {total_bytes} bytes) as {len(fragments)} fragments")
        
        return WriteResult(
            result=fragments,
            bytes_written=total_bytes,
            rows_written=total_rows,
        )

    def finalize(self, write_results: list[WriteResult[list[lance.FragmentMetadata]]]) -> MicroPartition:
        """Commits the fragments to the Lance dataset. Returns a DataFrame with the stats of the dataset."""
        lance = self._import_lance()

        fragments = list(chain.from_iterable(write_result.result for write_result in write_results))

        if self._mode == "create" or self._mode == "overwrite":
            operation = lance.LanceOperation.Overwrite(self._pyarrow_schema, fragments)
        elif self._mode == "append":
            operation = lance.LanceOperation.Append(fragments)
        else:
            raise ValueError(f"Unsupported mode: {self._mode}")

        try:
            dataset = lance.LanceDataset.commit(
                self._table_uri,
                operation,
                read_version=self._version,
                storage_options=self._storage_options,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to commit {len(fragments)} fragments to Lance dataset: {e}") from e
        
        stats = dataset.stats.dataset_stats()

        tbl = MicroPartition.from_pydict(
            {
                "num_fragments": pa.array([stats["num_fragments"]], type=pa.int64()),
                "num_deleted_rows": pa.array([stats["num_deleted_rows"]], type=pa.int64()),
                "num_small_files": pa.array([stats["num_small_files"]], type=pa.int64()),
                "version": pa.array([dataset.version], type=pa.int64()),
            }
        )
        
        logger.info(f"Finalized Lance dataset: {stats['num_fragments']} fragments, "
                   f"version {dataset.version}")
        
        return tbl
