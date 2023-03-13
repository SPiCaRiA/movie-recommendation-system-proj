from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .typing import IntArray


def loss_mae(truth_arr: IntArray | list[int],
             pred_arr: IntArray | list[int]) -> float:
    if isinstance(truth_arr, list):
        truth_arr = np.array(truth_arr)

    if isinstance(pred_arr, list):
        pred_arr = np.array(pred_arr)

    return float(np.mean(np.abs(truth_arr - pred_arr)))


def loss_mse(truth_arr: IntArray | list[int],
             pred_arr: IntArray | list[int]) -> float:
    if isinstance(truth_arr, list):
        truth_arr = np.array(truth_arr)

    if isinstance(pred_arr, list):
        pred_arr = np.array(pred_arr)

    return (1 / len(truth_arr)) * np.sum((truth_arr - pred_arr)**2)


def loss_rmse(truth_arr: IntArray | list[int],
              pred_arr: IntArray | list[int]) -> float:
    return float(np.sqrt(loss_mse(truth_arr, pred_arr)))
