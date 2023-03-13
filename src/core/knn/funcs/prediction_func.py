from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ....typing import IntMaskedArray
from ...shared.prediction import prediction_func

if TYPE_CHECKING:
    from ...shared.prediction import PredictionGenerator


@prediction_func
def weighted_average(col: int, _active_row: IntMaskedArray,
                     _active_arr: IntMaskedArray) -> PredictionGenerator:
    total: float = 0
    total_weight: float = 0
    res: float = 0
    accessor = lambda: res

    try:
        while True:
            weight, cur_row = yield accessor
            total_weight += weight
            total += cur_row[col] * weight
    except GeneratorExit:
        if total_weight != 0:
            res = total / total_weight


@prediction_func
def diff_weighted_average(col: int, active_row: IntMaskedArray,
                          _active_user: IntMaskedArray) -> PredictionGenerator:
    total: float = 0
    total_weight: float = 0
    res: float = 0
    accessor = lambda: res

    try:
        while True:
            weight, cur_row = yield accessor
            total_weight += abs(weight)
            total += (cur_row[col] - np.ma.mean(cur_row)) * weight
    except GeneratorExit:
        if total_weight != 0:
            res = total / total_weight + np.ma.mean(active_row)


@prediction_func
def adj_diff_weighted_average(
        col: int, _active_row: IntMaskedArray,
        active_user: IntMaskedArray) -> PredictionGenerator:
    total: float = 0
    total_weight: float = 0
    res: float = 0
    accessor = lambda: res

    try:
        while True:
            weight, cur_row = yield accessor
            total_weight += abs(weight)
            total += (cur_row[col] - np.ma.mean(active_user)) * weight
    except GeneratorExit:
        if total_weight != 0:
            res = total / total_weight + np.ma.mean(active_user)
