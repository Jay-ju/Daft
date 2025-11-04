# Copyright (c) Beijing Volcano Engine Technology Ltd.
from __future__ import annotations

import logging
import math
from typing import Any

from daft.dependencies import pa
from daft.las.functions.types import Operator
from daft.las.functions.utils.common_utils import tracking_usage

logger = logging.getLogger(__name__)


class TimestampsMerge(Operator):
    """**时间戳合并算子 - 秒级区间的规范化、合并与切分**

    **核心能力**
    - 规范化与排序：统一输入格式为 (start, end) 浮点秒并校验合法性。
    - 预合并小间隙：合并重叠或间隙小于等于阈值的相邻片段（pre_merge_gap_seconds）。
    - 最大静默优先切分：在最长静默处优先切分，保证每段跨度不超过 max_span_seconds。
    - 强制切块（可选）：对超长片段按固定窗口切分，确保每段长度不超过上限（enforce_chunking）。

    **推荐实践**
    - 当 VAD 输出存在短静默且希望保留语义连续性时，优先使用“最大静默优先切分”而非简单顺序切块。
    - 若业务要求硬性限制片段长度，开启 enforce_chunking 以获得连续窗口切分效果。
    - 将 pre_merge_gap_seconds 设置为略大于噪声静默的典型时长（例如 0.2–1.0 秒）。

    **输出格式**
    - 每行返回 List[List[float]]，元素为 [start, end]（单位：秒），按起点升序；输出列类型为 List[List[float32]]。

    **处理流程**
    1. 规范化：加上统一偏移 start_time、排序并校验合法性。
    2. 预合并：合并重叠与小间隙（≤ pre_merge_gap_seconds）的片段。
    3. 最大静默优先切分：在最长静默处分割，确保每段跨度 ≤ max_span_seconds；对单片段时间范围不做≤ max_span_seconds的要求。
    4. （可选）强制切块：当 enforce_chunking=True 时，按固定窗口连续切分确保时间长度 ≤ max_span_seconds。
    """  # noqa: D415

    def __init__(
        self,
        start_time: float = 0.0,
        pre_merge_gap_seconds: float = 0.0,
        max_span_seconds: float = 20.0,
        enforce_chunking: bool = False,
        **kwargs: Any,
    ) -> None:
        """初始化 TimestampsMerge 算子的参数.

        Args:
            start_time: 应用于每个时间戳的起始偏移（秒）。
                默认值：0
            pre_merge_gap_seconds: 微小间隙预合并的阈值（秒）。
                当相邻片段间隔小于等于此值时，会合并为一个更大的片段。
                默认值：0
            max_span_seconds: 最大允许的片段合并跨度（秒）。
                默认值：20
            enforce_chunking: 是否强制对超长片段进行连续切块。
                当为 True 时，会按 max_span_seconds 长度进行连续切块。
                默认值：False
        """
        super().__init__(**kwargs)
        if max_span_seconds <= 0:
            raise ValueError("max_span_seconds must be positive")
        if pre_merge_gap_seconds < 0:
            raise ValueError("pre_merge_gap_seconds cannot be negative")

        self.start_time = float(start_time)
        self.pre_merge_gap_seconds = float(pre_merge_gap_seconds)
        self.max_span_seconds = float(max_span_seconds)
        self.enforce_chunking = bool(enforce_chunking)

        tracking_usage(op=self.__class__.__name__)
        logger.debug(
            "TimestampsMerge init: start_time=%.3f, pre_merge_gap_seconds=%.3f, max_span_seconds=%.3f, enforce_chunking=%s",
            self.start_time,
            self.pre_merge_gap_seconds,
            self.max_span_seconds,
            self.enforce_chunking,
        )

    def transform(self, timestamps: pa.Array) -> pa.Array:
        """批量处理时间戳列，输出按约束合并后的区间.

        Args:
            timestamps: 其中每个元素为二维浮点数列表（单位：秒），形如 [[start, end], ...]。

        Returns:
            每个元素为合并后的二维浮点数列表（单位：秒），类型为 List[List[float32]]；非法行返回 None。

        注：处理流程为“规范化 → 预合并小间隙 → 最大静默优先切分 →（可选）强制切块”。
        """
        rows = timestamps.to_pylist()
        logger.debug("TimestampsMerge transform: input row count=%d", len(rows))
        results: list[list[list[float]] | None] = []

        for idx, row in enumerate(rows):
            try:
                # 1) 规范化并排序
                segments = _normalize_row(row, start_time=self.start_time)
                if not segments:
                    results.append([])
                    logger.debug("row %d: empty after normalization, returning empty list", idx)
                    continue

                # 2) 预合并重叠与小间隙
                merged_segments = _merge_small_gaps(segments, pre_gap=self.pre_merge_gap_seconds)
                if not merged_segments:
                    results.append([])
                    logger.debug("row %d: empty after pre-merge, returning empty list", idx)
                    continue
                logger.debug("row %d: segments after pre-merge=%d", idx, len(merged_segments))

                # 3) 最大静默优先切分（迭代栈），保证每段跨度 ≤ max_span_seconds
                final_segments = _segment_speech_merge(merged_segments, max_span_seconds=self.max_span_seconds)
                logger.debug("row %d: segments after largest-gap-first=%d", idx, len(final_segments))

                # 4) （可选）连续窗口强制切块
                if self.enforce_chunking:
                    before = len(final_segments)
                    chunked = _chunk_if_needed(final_segments, self.max_span_seconds)
                    after = len(chunked)
                    # 保持升序
                    chunked.sort(key=lambda x: (x[0], x[1]))
                    logger.info("row %d: enforce chunking enabled, segments before=%d, after=%d", idx, before, after)
                    final_segments = chunked

                # 5) 转换为 List[List[float]]，输出 float32 类型由 Arrow cast
                out_row: list[list[float]] = [[float(s), float(e)] for (s, e) in final_segments]
                results.append(out_row)
            except Exception as e:
                logger.warning("TimestampsMerge: row %d failed, returning None. Reason: %s", idx, e)
                results.append(None)

        return pa.array(results, type=self.__return_column_type__())

    @staticmethod
    def __return_column_type__() -> pa.DataType:
        return pa.list_(pa.list_(pa.float32()))


def _normalize_row(row: Any, start_time: float) -> list[tuple[float, float]]:
    """将输入行规范为升序的 (start, end) 列表，并加上统一偏移.

    允许输入格式：
    - List[List[float]] 或 List[Tuple[float, float]]，每个子元素为 [start, end]

    规则：
    - 对每个片段加上 start_time 偏移（单位：秒）
    - 校验 start/end 为有限数值，且 start ≤ end
    - 按 start 升序（若相等按 end 升序）
    """
    if row is None:
        return []
    if not isinstance(row, (list, tuple)):
        raise TypeError(f"expected list or tuple, got {type(row)!r}")

    normalized: list[tuple[float, float]] = []
    for i, item in enumerate(row):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"element {i} is not a 2-item list/tuple: {item!r}")
        s_raw, e_raw = item[0], item[1]
        try:
            s = float(s_raw) + float(start_time)
            e = float(e_raw) + float(start_time)
        except Exception as exc:
            raise ValueError(f"element {i} cannot be converted to float: {item!r}") from exc

        if not (math.isfinite(s) and math.isfinite(e)):
            raise ValueError(f"element {i} contains non-finite values: {item!r}")
        if s > e:
            raise ValueError(f"element {i} has start > end: {item!r}")

        normalized.append((s, e))

    normalized.sort(key=lambda x: (x[0], x[1]))
    return normalized


def _merge_small_gaps(segments: list[tuple[float, float]], pre_gap: float) -> list[tuple[float, float]]:
    """合并重叠与小间隙片段，返回不重叠且有序的区间列表.

    合并逻辑：
    - 输入应为按 start 升序的片段序列（由 _normalize_row 保证）
    - 若下一个片段的起点 <= 当前片段的终点（重叠）或间隙 <= pre_gap（小间隙），则合并
    - 否则输出当前片段并开启新的片段
    """
    if not segments:
        return []

    merged: list[tuple[float, float]] = []
    cur_s, cur_e = segments[0]

    for i in range(1, len(segments)):
        ns, ne = segments[i]
        # 间隙大小（ns - cur_e）；重叠时 gap <= 0
        gap = ns - cur_e
        if gap <= pre_gap:
            # 重叠或间隙不超过阈值，进行合并
            cur_e = max(cur_e, ne)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = ns, ne
    merged.append((cur_s, cur_e))
    return merged


def _segment_speech_merge(segments: list[tuple[float, float]], max_span_seconds: float) -> list[tuple[float, float]]:
    """对已预合并的片段进行“最大静默优先”的迭代切分，保证每段跨度不超过上限.

    行为说明：
    - 预计算相邻片段之间的静默时长（gaps），保证 gaps[i] = segments[i+1].start - segments[i].end ≥ 0
    - 使用显式栈处理索引范围 [s_idx, e_idx]：
      * 若 s_idx == e_idx 或当前范围合并后的跨度 ≤ max_span_seconds，则接受该范围；
      * 否则在 [s_idx, e_idx) 内找到最大的间隙位置进行二分，将左右范围继续压栈处理。
    - 输出按起点升序排列的 (start, end) 列表。

    注意：当范围仅包含单个片段时（s_idx == e_idx），无论其长度是否超过上限，都会被接受（与历史递归实现保持一致）。
    """
    if not segments:
        return []

    n = len(segments)
    # 预计算相邻间隙
    gaps: list[float] = []
    for i in range(n - 1):
        gap = segments[i + 1][0] - segments[i][1]
        gaps.append(max(0.0, float(gap)))

    result_ranges: list[tuple[int, int]] = []
    stack: list[tuple[int, int]] = [(0, n - 1)]

    while stack:
        s_idx, e_idx = stack.pop()
        if s_idx > e_idx:
            continue

        span = segments[e_idx][1] - segments[s_idx][0]
        if s_idx == e_idx or span <= max_span_seconds:
            result_ranges.append((s_idx, e_idx))
            continue

        # 在 [s_idx, e_idx) 内寻找最大的间隙位置
        if e_idx - s_idx >= 1:
            sub_start = s_idx
            sub_end_excl = e_idx
            max_gap = -1.0
            max_gap_index = s_idx
            for gi in range(sub_start, sub_end_excl):
                gval = gaps[gi]
                if gval > max_gap:
                    max_gap = gval
                    max_gap_index = gi
            split_index = max_gap_index

            # 压入左右两段
            stack.append((s_idx, split_index))
            stack.append((split_index + 1, e_idx))
        else:
            result_ranges.append((s_idx, e_idx))

    # 将索引范围转为具体区间
    result: list[tuple[float, float]] = [(segments[s][0], segments[e][1]) for (s, e) in result_ranges]
    result.sort(key=lambda x: (x[0], x[1]))
    return result


def _chunk_if_needed(segments: list[tuple[float, float]], max_span: float) -> list[tuple[float, float]]:
    """按照 max_span 对片段进行强制切块（连续窗口）.

    - 对于每个片段 [s, e]，当 (e - s) > max_span 时，切分为 [s, s+max_span], [s+max_span, s+2*max_span], ...，直至覆盖到 e。
    - 最后一个窗口可能小于 max_span，但不会超过原片段的 e。
    - 若 max_span <= 0 或不可用，则直接返回原始片段。
    """
    out: list[tuple[float, float]] = []
    try:
        span = float(max_span)
    except Exception:
        span = 0.0
    if span <= 0.0:
        return segments

    for s, e in segments:
        length = e - s
        if length <= span:
            out.append((s, e))
            continue
        cursor = s
        while cursor < e:
            next_end = cursor + span
            if next_end >= e:
                out.append((cursor, e))
                break
            out.append((cursor, next_end))
            cursor = next_end
    return out
