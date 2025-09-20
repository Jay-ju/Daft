
# ruff: noqa: I002
# isort: dont-add-import: from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Optional, Union

from daft.daft import CountMode, PyExpr, PyPartitionField, PyPushdowns, PyRecordBatch, ScanTask
from daft.dependencies import pa
from daft.expressions import Expression
from daft.io.scan import ScanOperator
from daft.logical.schema import Schema
from daft.recordbatch import RecordBatch

from ..pushdowns import SupportsPushdownFilters

if TYPE_CHECKING:
    import lance

logger = logging.getLogger(__name__)


# Scan Strategy Abstract Base Classes
class ScanStrategy(ABC):
    """Abstract base class for different Lance scan strategies."""
    
    @abstractmethod
    def create_scan_tasks(
        self,
        operator: 'LanceDBScanOperator',
        pushdowns: PyPushdowns,
        required_columns: Optional[list[str]]
    ) -> Iterator[ScanTask]:
        """Create scan tasks using this strategy."""
        pass


class SingleFragmentStrategy(ScanStrategy):
    """Strategy for processing each fragment individually."""
    
    def create_scan_tasks(
        self,
        operator: 'LanceDBScanOperator',
        pushdowns: PyPushdowns,
        required_columns: Optional[list[str]]
    ) -> Iterator[ScanTask]:
        fragments = operator._ds.get_fragments()
        pushed_expr = operator._combine_filters_to_arrow()
        
        for fragment in fragments:
            yield operator._create_scan_task(
                fragment_ids=[fragment.fragment_id],
                required_columns=required_columns,
                pushdowns=pushdowns,
                filter_expr=pushed_expr,
                limit=pushdowns.limit
            )


class GroupedFragmentStrategy(ScanStrategy):
    """Strategy for processing fragments in groups for better parallelism."""
    
    def __init__(self, parallelism: int):
        self.parallelism = parallelism
    
    def create_scan_tasks(
        self,
        operator: 'LanceDBScanOperator',
        pushdowns: PyPushdowns,
        required_columns: Optional[list[str]]
    ) -> Iterator[ScanTask]:
        fragments = operator._ds.get_fragments()
        fragment_groups = operator._group_fragments(fragments, self.parallelism)
        pushed_expr = operator._combine_filters_to_arrow()
        
        for fragment_group in fragment_groups:
            fragment_ids = [fragment.fragment_id for fragment in fragment_group]
            yield operator._create_scan_task(
                fragment_ids=fragment_ids,
                required_columns=required_columns,
                pushdowns=pushdowns,
                filter_expr=pushed_expr,
                limit=pushdowns.limit
            )


class LimitOptimizedStrategy(ScanStrategy):
    """Strategy optimized for limit pushdown with no filters."""
    
    def create_scan_tasks(
        self,
        operator: 'LanceDBScanOperator',
        pushdowns: PyPushdowns,
        required_columns: Optional[list[str]]
    ) -> Iterator[ScanTask]:
        assert operator._pushed_filters is None, "Expected no filters when using limit optimized strategy"
        assert pushdowns.limit is not None, "Expected a limit when using limit optimized strategy"
        
        fragments = operator._ds.get_fragments()
        remaining_limit = pushdowns.limit
        
        for fragment in fragments:
            if remaining_limit <= 0:
                break
            
            # Calculate effective rows using fragment.count_rows()
            # This is not expensive because count_rows simply checks physical_rows - num_deletions when there are no filters
            # https://github.com/lancedb/lance/blob/v0.34.0/rust/lance/src/dataset/fragment.rs#L1049-L1055
            effective_rows = fragment.count_rows()
            
            if effective_rows > 0:
                rows_to_scan = min(remaining_limit, effective_rows)
                remaining_limit -= rows_to_scan
                
                yield operator._create_scan_task(
                    fragment_ids=[fragment.fragment_id],
                    required_columns=required_columns,
                    pushdowns=pushdowns,
                    filter_expr=None,
                    limit=rows_to_scan,
                    num_rows=rows_to_scan
                )


# TODO support fts and fast_search
def _lancedb_table_factory_function(
    ds: "lance.LanceDataset",
    fragment_ids: Optional[list[int]] = None,
    required_columns: Optional[list[str]] = None,
    filter: Optional["pa.compute.Expression"] = None,
    limit: Optional[int] = None,
) -> Iterator[PyRecordBatch]:
    fragments = [ds.get_fragment(id) for id in (fragment_ids or [])]
    if not fragments:
        raise RuntimeError(f"Unable to find lance fragments {fragment_ids}")
    scanner = ds.scanner(fragments=fragments, columns=required_columns, filter=filter, limit=limit)
    return (RecordBatch.from_arrow_record_batches([rb], rb.schema)._recordbatch for rb in scanner.to_batches())


def _lancedb_count_result_function(
    ds: "lance.LanceDataset",
    required_column: str,
    filter: Optional["pa.compute.Expression"] = None,
) -> Iterator[PyRecordBatch]:
    """Use LanceDB's API to count rows and return a record batch with the count result."""
    logger.debug("Using metadata for counting all rows")
    count = ds.count_rows(filter=filter)

    arrow_schema = pa.schema([pa.field(required_column, pa.uint64())])
    arrow_array = pa.array([count], type=pa.uint64())
    arrow_batch = pa.RecordBatch.from_arrays([arrow_array], [required_column])
    result_batch = RecordBatch.from_arrow_record_batches([arrow_batch], arrow_schema)._recordbatch
    return (result_batch for _ in [1])


class LanceDBScanOperator(ScanOperator, SupportsPushdownFilters):
    def __init__(self, ds: "lance.LanceDataset", parallelism: Optional[int] = None, fragment_group_size: Optional[int] = None):
        """Initialize LanceDB scan operator.
        
        Args:
            ds: Lance dataset to scan
            parallelism: Number of fragments to group together in a single scan task.
                        If None or <= 1, each fragment will be processed individually.
            fragment_group_size: Deprecated parameter name for parallelism.
                               Use 'parallelism' instead.
        """
        self._ds = ds
        self._pushed_filters: Union[list[PyExpr], None] = None
        
        # Handle parameter naming with backward compatibility
        if fragment_group_size is not None and parallelism is not None:
            raise ValueError("Cannot specify both 'parallelism' and 'fragment_group_size'. Use 'parallelism' only.")
        
        if fragment_group_size is not None:
            warnings.warn(
                "Parameter 'fragment_group_size' is deprecated. Use 'parallelism' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            self._parallelism = fragment_group_size
        else:
            self._parallelism = parallelism
        
        # Keep the old attribute for backward compatibility
        self._fragment_group_size = self._parallelism

    def name(self) -> str:
        return "LanceDBScanOperator"

    def display_name(self) -> str:
        return f"LanceDBScanOperator({self._ds.uri})"

    def schema(self) -> Schema:
        return Schema.from_pyarrow_schema(self._ds.schema)

    def partitioning_keys(self) -> list[PyPartitionField]:
        return []

    def can_absorb_filter(self) -> bool:
        return isinstance(self, SupportsPushdownFilters)

    def can_absorb_limit(self) -> bool:
        return True

    def can_absorb_select(self) -> bool:
        return True

    def supports_count_pushdown(self) -> bool:
        """Returns whether this scan operator supports count pushdown."""
        return True

    def supported_count_modes(self) -> list[CountMode]:
        """Returns the count modes supported by this scan operator."""
        return [CountMode.All]

    def as_pushdown_filter(self) -> Union[SupportsPushdownFilters, None]:
        return self

    def multiline_display(self) -> list[str]:
        return [
            self.display_name(),
            f"Schema = {self.schema()}",
        ]

    def push_filters(self, filters: list[PyExpr]) -> tuple[list[PyExpr], list[PyExpr]]:
        pushed = []
        remaining = []

        for expr in filters:
            try:
                Expression._from_pyexpr(expr).to_arrow_expr()
                pushed.append(expr)
            except NotImplementedError:
                remaining.append(expr)

        if pushed:
            self._pushed_filters = pushed
        else:
            self._pushed_filters = None

        return pushed, remaining

    def _create_scan_task(
        self,
        fragment_ids: list[int],
        required_columns: Optional[list[str]],
        pushdowns: PyPushdowns,
        num_rows: Optional[int] = None,
        filter_expr: Optional["pa.compute.Expression"] = None,
        limit: Optional[int] = None
    ) -> ScanTask:
        """Unified ScanTask creation function to eliminate code duplication.
        
        Args:
            fragment_ids: List of fragment IDs to include in this scan task
            required_columns: Columns to read from the fragments
            pushdowns: Pushdown operations to apply
            num_rows: Number of rows expected (for metadata)
            filter_expr: Arrow filter expression to apply
            limit: Row limit for this specific task
        
        Returns:
            Configured ScanTask for the given fragments
        """
        return ScanTask.python_factory_func_scan_task(
            module=_lancedb_table_factory_function.__module__,
            func_name=_lancedb_table_factory_function.__name__,
            func_args=(self._ds, fragment_ids, required_columns, filter_expr, limit),
            schema=self.schema()._schema,
            num_rows=num_rows,
            size_bytes=None,
            pushdowns=pushdowns,
            stats=None,
        )

    def _group_fragments(self, fragments: list, parallelism: int) -> list[list]:
        """Group fragments into batches for parallel processing.
        
        Args:
            fragments: List of fragments to group
            parallelism: Number of fragments per group
        
        Returns:
            List of fragment groups
        """
        if parallelism <= 1:
            return [[fragment] for fragment in fragments]
        
        groups = []
        current_group = []
        
        for fragment in fragments:
            current_group.append(fragment)
            if len(current_group) >= parallelism:
                groups.append(current_group)
                current_group = []
        
        # Add the last group if it has any fragments
        if current_group:
            groups.append(current_group)
        
        return groups

    def _get_scan_strategy(self, pushdowns: PyPushdowns) -> ScanStrategy:
        """Select the appropriate scan strategy based on pushdowns and configuration.
        
        Args:
            pushdowns: Pushdown operations to consider
        
        Returns:
            Appropriate ScanStrategy instance
        """
        # Use limit optimized strategy for limit pushdown with no filters
        if pushdowns.limit is not None and self._pushed_filters is None:
            return LimitOptimizedStrategy()
        
        # Use grouped strategy if parallelism is configured
        if self._parallelism is not None and self._parallelism > 1:
            return GroupedFragmentStrategy(self._parallelism)
        
        # Default to single fragment strategy
        return SingleFragmentStrategy()

    def to_scan_tasks(self, pushdowns: PyPushdowns) -> Iterator[ScanTask]:
        """Generate scan tasks using the appropriate strategy.
        
        This method has been refactored to use the strategy pattern,
        eliminating complex conditional logic and code duplication.
        """
        required_columns: Optional[list[str]]
        if pushdowns.columns is None:
            required_columns = None
        else:
            filter_required_column_names = pushdowns.filter_required_column_names()
            required_columns = list(
                set(
                    pushdowns.columns
                    if filter_required_column_names is None
                    else pushdowns.columns + filter_required_column_names
                )
            )

        # Check if there is a count aggregation pushdown
        if (
            pushdowns.aggregation is not None
            and pushdowns.aggregation_count_mode() is not None
            and pushdowns.aggregation_required_column_names()
        ):
            count_mode = pushdowns.aggregation_count_mode()
            fields = pushdowns.aggregation_required_column_names()

            if count_mode not in self.supported_count_modes():
                logger.warning(
                    "Count mode %s is not supported for pushdown, falling back to regular scan strategy",
                    count_mode,
                )
                strategy = self._get_scan_strategy(pushdowns)
                yield from strategy.create_scan_tasks(self, pushdowns, required_columns)
                return

            filters = self._combine_filters_to_arrow()

            new_schema = Schema.from_pyarrow_schema(pa.schema([pa.field(fields[0], pa.uint64())]))
            yield ScanTask.python_factory_func_scan_task(
                module=_lancedb_count_result_function.__module__,
                func_name=_lancedb_count_result_function.__name__,
                func_args=(self._ds, fields[0], filters),
                schema=new_schema._schema,
                num_rows=1,
                size_bytes=None,
                pushdowns=pushdowns,
                stats=None,
            )
        else:
            # Use strategy pattern to determine the appropriate scan approach
            strategy = self._get_scan_strategy(pushdowns)
            yield from strategy.create_scan_tasks(self, pushdowns, required_columns)

    # Legacy methods kept for backward compatibility
    # These methods are now deprecated and replaced by the strategy pattern
    
    def _create_scan_tasks_with_limit_and_no_filters(
        self, pushdowns: PyPushdowns, required_columns: Optional[list[str]]
    ) -> Iterator[ScanTask]:
        """Deprecated: Use LimitOptimizedStrategy instead."""
        warnings.warn(
            "_create_scan_tasks_with_limit_and_no_filters is deprecated. "
            "The method now uses LimitOptimizedStrategy internally.",
            DeprecationWarning,
            stacklevel=2
        )
        strategy = LimitOptimizedStrategy()
        yield from strategy.create_scan_tasks(self, pushdowns, required_columns)

    def _create_regular_scan_tasks(
        self, pushdowns: PyPushdowns, required_columns: Optional[list[str]]
    ) -> Iterator[ScanTask]:
        """Deprecated: Use appropriate strategy instead."""
        warnings.warn(
            "_create_regular_scan_tasks is deprecated. "
            "The method now uses strategy pattern internally.",
            DeprecationWarning,
            stacklevel=2
        )
        strategy = self._get_scan_strategy(pushdowns)
        yield from strategy.create_scan_tasks(self, pushdowns, required_columns)

    def _combine_filters_to_arrow(self) -> Optional["pa.compute.Expression"]:
        """Combine pushed filters into a single Arrow expression.
        
        Returns:
            Combined Arrow filter expression or None if no filters
        """
        if self._pushed_filters is not None:
            combined_filter = self._pushed_filters[0]
            for filter_expr in self._pushed_filters[1:]:
                combined_filter = combined_filter & filter_expr
            return Expression._from_pyexpr(combined_filter).to_arrow_expr()
        return None
