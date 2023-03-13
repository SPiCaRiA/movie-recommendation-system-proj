from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ....typing import NanSimilarityMatrix, RatingMatrix


def cosine_similarity(r1: RatingMatrix,
                      r2: RatingMatrix) -> NanSimilarityMatrix:
    '''
    r1: u x i matrix
    r2: v x i matrix
    return: matrix of size u x v, with similarities between rows in r1 and r2.
    '''
    dot_m = np.ma.dot(r1, r2.T)

    # Add new axes to enable broadcasting when manually calculating norms of r1
    # and r2.T with the union mask applied.
    r1_na = r1[..., None]
    r2t_na = r2.T[None, ...]

    mask_m = r1_na.mask | r2t_na.mask
    r1_broadcast = np.broadcast_to(r1_na**2, mask_m.shape, subok=True)
    r2t_broadcast = np.broadcast_to(r2t_na**2, mask_m.shape, subok=True)

    # The broadcasted arrays are ensured to be MaskedArrays since we set the
    # subok flag, but the type checker has no clue about this.
    r1_broadcast.mask = mask_m    # type: ignore
    r2t_broadcast.mask = mask_m    # type: ignore

    # Note: if the intersection between two rows has length 1, we set sim as 0
    #       to avoid false positive of cosine similarity.
    #       This is done by setting zero for norm, which results in zero-division
    #       in res, and fill all the invalid value in res as 0 before returning.
    norm_r1 = np.apply_along_axis(
        lambda c: 0 if c.count() == 1 else np.ma.sum(c), 1, r1_broadcast)
    norm_r2 = np.apply_along_axis(
        lambda c: 0 if c.count() == 1 else np.ma.sum(c), 1, r2t_broadcast)

    prod_norm = np.ma.sqrt(norm_r1 * norm_r2)

    return dot_m / prod_norm


def pearson_correlation(r1: RatingMatrix,
                        r2: RatingMatrix) -> NanSimilarityMatrix:
    '''
    r1: u x i matrix
    r2: v x i matrix
    return: matrix of size u x v, with similarities between rows in r1 and r2.
    '''
    r1_mean = np.ma.mean(r1, axis=1)[:, None]
    r2_mean = np.ma.mean(r2, axis=1)[:, None]
    return cosine_similarity(r1 - r1_mean, r2 - r2_mean)


def adjusted_cosine_similarity(r1: RatingMatrix,
                               r2: RatingMatrix) -> NanSimilarityMatrix:
    '''
    The pearson correlation for *item-based CF* only that uses the average rating
    of the users instead of the items.

    r1: i x u matrix
    r2: i x u matrix
    return: matrix of size i x i, with similarities between rows in r1 and r2.
    '''
    # Note that for item-based CF, r1 and r2 should be same (i.e. the known
    # rating matrix), so they can be applied with the same user mean matrix.
    user_mean = np.ma.mean(r1, axis=0)
    return cosine_similarity(r1 - user_mean, r2 - user_mean)
