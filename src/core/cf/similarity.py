from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...typing import Similarity

if TYPE_CHECKING:
    from ...typing import FloatMatrix, FSimilarity, RatingMatrix, SupportMatrix


def similarity_matrix(
        r1: RatingMatrix,
        r2: RatingMatrix,
        similarity_func: FSimilarity,
        fill_value: int,
        weights: tuple[FloatMatrix, FloatMatrix] | None = None) -> Similarity:
    if weights is not None:
        weight1, weight2 = weights
        sim_m = similarity_func(r1 * weight1, r2 * weight2).filled(fill_value)
    else:
        sim_m = similarity_func(r1, r2).filled(fill_value)
    return Similarity(sim_m)


def support_matrix(r1: RatingMatrix, r2: RatingMatrix) -> SupportMatrix:
    '''
    r1: m x k matrix
    r2: n x k matrix
    return: matrix of size m x n of support(m, n); support(m, n) is defined by
        user-based: number of items user m and n both rated
        item-based: number of users item m and n are both rated
    '''
    r1_na = r1[..., None]
    r2t_na = r2.T[None, ...]
    mask_m = np.transpose(r1_na.mask | r2t_na.mask, (0, 2, 1))
    return r1.shape[1] - np.count_nonzero(mask_m, axis=2)