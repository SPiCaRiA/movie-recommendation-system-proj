from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .type_aliases import NpFloat, NpInt

if TYPE_CHECKING:
    import numpy.typing as npt

    from .type_aliases import (
        FloatArray,
        FloatMatrix,
        IntArray,
        IntMaskedArray,
        IntMaskedMatrix,
        IntMatrix,
    )


def int_array(len_or_val: int | np.int_ | list[int] | IntArray) -> IntArray:
    if isinstance(len_or_val, int):
        return np.zeros(len_or_val, dtype=NpInt)
    else:
        return np.array(len_or_val, dtype=NpInt)


def int_matrix(
    shape_or_val: tuple[int | np.int_, int | np.int_] | list[list[int]] |
    IntMatrix
) -> IntMatrix:
    if isinstance(shape_or_val, tuple):
        return np.zeros(shape_or_val, dtype=NpInt)
    else:
        return np.array(shape_or_val, dtype=NpInt)


def float_array(
        len_or_val: int | np.int_ | list[float] | FloatArray) -> FloatArray:
    if isinstance(len_or_val, int):
        return np.zeros(len_or_val, dtype=NpFloat)
    else:
        return np.array(len_or_val, dtype=NpFloat)


def float_matrix(
    shape_or_val: tuple[int | np.int_, int | np.int_] | list[list[int]] |
    FloatMatrix
) -> FloatMatrix:
    if isinstance(shape_or_val, tuple):
        return np.zeros(shape_or_val, dtype=NpFloat)
    else:
        return np.array(shape_or_val, dtype=NpFloat)


def int_masked_array(
        condition: bool | npt.NDArray[np.bool_],
        len_or_val: int | np.int_ | list[int] | IntArray) -> IntMaskedArray:
    return np.ma.masked_where(condition, int_array(len_or_val))


def int_masked_matrix(
    condition: bool | npt.NDArray[np.bool_],
    shape_or_val: tuple[int | np.int_, int | np.int_] | list[list[int]] |
    IntMatrix
) -> IntMaskedMatrix:
    return np.ma.masked_where(condition, int_matrix(shape_or_val))
