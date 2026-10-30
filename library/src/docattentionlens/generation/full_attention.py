"""Full prefill attention for generator captures, stored losslessly and read back as causal squares.

The eager decoder materializes a dense (heads, n, n) post-softmax matrix per layer during prefill.
An opt-in capture keeps its causal lower triangle as raw BF16 bit patterns (uint16), which is exact
for the BF16 weights the model computed. Row i holds targets 0..i, rows concatenated in order.

A square for a selected source s (s + 1 positions) is rebuilt from the prefill triangle for rows
below n, and from the retained prediction-step rows for decode positions n..s: step k >= 1 is the
row of position n + k - 1. Positions after the last retained decode row are never synthesized.
"""
import numpy as np

LAYOUT = "bf16_lower_triangle_rows_v1"


def triangle_length(n: int) -> int:
    return n * (n + 1) // 2


def array_name(layer: int) -> str:
    return f"full_l{layer}"


def bf16_bits(values) -> np.ndarray:
    """Raw BF16 bit patterns of float32 values that are exactly BF16; refuses anything else."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    if np.any(bits & 0xFFFF):
        raise ValueError("Attention values are not exact BF16; refusing lossy storage")
    return (bits >> 16).astype(np.uint16)


def from_bits(bits) -> np.ndarray:
    """Exact float32 values of stored BF16 bit patterns."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def pack(square) -> np.ndarray:
    """(heads, n, n) causal square -> (heads, n(n+1)/2) uint16 lower triangle. Future keys must be zero."""
    square = np.asarray(square, dtype=np.float32)
    if square.ndim != 3 or square.shape[1] != square.shape[2]:
        raise ValueError("Expected a (heads, n, n) attention square")
    n = square.shape[1]
    rows, cols = np.tril_indices(n)
    if np.any(square[:, ~np.tri(n, dtype=bool)] != 0):
        raise ValueError("Causal prefill attention includes future keys")
    return bf16_bits(square[:, rows, cols])


def unpack(packed, n: int, upto: int | None = None) -> np.ndarray:
    """uint16 triangle -> float32 (heads, m, m) square of the first m = upto (default n) positions."""
    packed = np.asarray(packed)
    m = n if upto is None else upto
    if packed.dtype != np.uint16 or packed.ndim != 2 or packed.shape[1] != triangle_length(n) or not 0 < m <= n:
        raise ValueError("Full attention triangle differs from its declared axes")
    rows, cols = np.tril_indices(m)
    square = np.zeros((packed.shape[0], m, m), dtype=np.float32)
    # Rows are stored in order, so the first m rows are the first m(m+1)/2 entries.
    square[:, rows, cols] = from_bits(packed[:, :triangle_length(m)])
    return square


def square(packed, n: int, source: int, decode_rows=()) -> np.ndarray:
    """Causal (heads, source+1, source+1) square for one layer.

    `decode_rows[k-1]` is the retained (heads, n+k) row of position n+k-1 for prediction step k >= 1,
    already sliced to this layer. Every row below the source must be present.
    """
    if not 0 <= source < n + len(decode_rows):
        raise ValueError("Selected source is outside the retained full-attention rows")
    m = source + 1
    if m <= n:
        return unpack(packed, n, m)
    result = np.zeros((packed.shape[0], m, m), dtype=np.float32)
    result[:, :n, :n] = unpack(packed, n)
    for k in range(1, m - n + 1):
        row = np.asarray(decode_rows[k - 1], dtype=np.float32)
        if row.shape != (packed.shape[0], n + k):
            raise ValueError("Retained decode row differs from the full-attention axes")
        result[:, n + k - 1, :n + k] = row
    return result


# A prefill position's own row, read from the triangle, stands in for a retained source row.
PREFILL_ROW = "prefill_row_"


def prefill_row_name(source: int) -> str:
    return f"{PREFILL_ROW}{source}"


def prefill_row_index(name: str) -> int | None:
    if not name.startswith(PREFILL_ROW) or not name[len(PREFILL_ROW):].isdigit():
        return None
    return int(name[len(PREFILL_ROW):])


def row(packed, n: int, source: int) -> np.ndarray:
    """Exact float32 (heads, source+1) row of one prefill position, without unpacking the square."""
    packed = np.asarray(packed)
    if packed.dtype != np.uint16 or packed.ndim != 2 or packed.shape[1] != triangle_length(n) or not 0 <= source < n:
        raise ValueError("Prefill row is outside the saved full-attention triangle")
    start = triangle_length(source)
    return from_bits(packed[:, start:start + source + 1])
