from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from .type_aliases import IntArray

Checker = Callable[[Any], bool]


def check(obj: Any, *checkers: Checker) -> bool:
    for checker in checkers:
        if not checker(obj):
            return False
    return True


def assertion(obj: Any, *checkers: Checker) -> None:
    assert check(obj, *checkers)


def RefinementError(target_type_name: str, obj: Any) -> TypeError:
    return TypeError(target_type_name, obj)


# --- Checkers ---


def check_int_array(obj: Any) -> bool:
    return isinstance(obj, np.ndarray) and len(
        obj.shape) == 1 and np.issubdtype(obj.dtype, np.integer)


# --- Refinement Functions ---


def refine_int_array(obj: Any) -> IntArray:
    if not check_int_array(obj):
        raise RefinementError('int array', obj)
    return obj
