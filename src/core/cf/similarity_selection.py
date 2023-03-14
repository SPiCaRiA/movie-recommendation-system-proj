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
        SupportMatrix,
    )


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


def indexed_support(row: int, support: SupportMatrix) -> IndexedSimArray:
    indices = np.arange(0, support.shape[1], dtype=support.dtype)
    return np.column_stack((indices, support[row].T))
