from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..prediction import prediction_func

if TYPE_CHECKING:
    from ....typing import IntMaskedArray, RatingMatrix
    from ..prediction import PredictionGenerator


@prediction_func
def weighted_average(col: int, _active_row: int, r: RatingMatrix,
                     _active_user: IntMaskedArray) -> PredictionGenerator:
    total: float = 0
    total_weight: float = 0
    res: float = 0
    accessor = lambda: res

    try:
        while True:
            weight, row = yield accessor
            total_weight += weight
            total += r[row, col] * weight
    except GeneratorExit:
        if total_weight != 0:
            res = total / total_weight


@prediction_func
def diff_weighted_average(col: int, active_row: int, r: RatingMatrix,
                          _active_user: IntMaskedArray) -> PredictionGenerator:
    total: float = 0
    total_weight: float = 0
    res: float = 0
    accessor = lambda: res

    try:
        while True:
            weight, row = yield accessor
            total_weight += abs(weight)
            total += (r[row, col] - np.ma.mean(r[row])) * weight
    except GeneratorExit:
        if total_weight != 0:
            res = total / total_weight + np.ma.mean(r[active_row])


@prediction_func
def adj_diff_weighted_average(
        col: int, _active_row: int, r: RatingMatrix,
        active_user: IntMaskedArray) -> PredictionGenerator:
    total: float = 0
    total_weight: float = 0
    res: float = 0
    accessor = lambda: res

    try:
        while True:
            weight, row = yield accessor
            total_weight += abs(weight)
            total += (r[row, col] - np.ma.mean(active_user)) * weight
    except GeneratorExit:
        if total_weight != 0:
            res = total / total_weight + np.ma.mean(active_user)


@prediction_func
def slope_one_weighted_average(
        _col: int, _active_row: int, _r: RatingMatrix,
        active_user: IntMaskedArray) -> PredictionGenerator:
    total: float = 0
    total_weight: float = 0
    res: float = 0
    accessor = lambda: res

    try:
        while True:
            weight, row = yield accessor
            total_weight += weight
            total += active_user[row] * weight
    except GeneratorExit:
        if total_weight != 0:
            res = total / total_weight
