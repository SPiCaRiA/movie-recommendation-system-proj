from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from ...typing import Similarity

if TYPE_CHECKING:
    from ...typing import (
        FIndexedSimMap,
        FloatArray,
        FloatMatrix,
        FPreSortSim,
        FSimilarity,
        IndexedSimArray,
        RatingMatrix,
    )


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


@cache
def indexed_desc_similarity(
        row: int,
        similarity: Similarity,
        pre_sort_sim: FPreSortSim | None = None,
        map_func: FIndexedSimMap | None = None) -> IndexedSimArray:
    sim_m = similarity.raw
    indices = np.arange(0, sim_m.shape[1], dtype=sim_m.dtype)
    active_sim: FloatArray = sim_m[row]

    sort_similarity_arr = active_sim.T
    if pre_sort_sim is not None:
        sort_similarity_arr = pre_sort_sim(sort_similarity_arr)

    indexed_active_sim = np.column_stack((indices, active_sim.T))
    reversed_sorted_index = sort_similarity_arr.argsort()[::-1]
    desc_sim = indexed_active_sim[reversed_sorted_index]

    if map_func is not None:
        return map_func(desc_sim)
    return desc_sim
