from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...typing import RatingMatrix


def support_matrix(r1: RatingMatrix, r2: RatingMatrix):
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
