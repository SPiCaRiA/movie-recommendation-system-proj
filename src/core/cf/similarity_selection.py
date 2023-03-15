from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...typing import (
        FIndexedSimMap,
        FloatArray,
        FPreSortSim,
        IndexedSimArray,
        Similarity,
        Support,
    )


@cache
def indexed_desc_similarity(
        sim_row: int,
        similarity: Similarity,
        pre_sort_sim: FPreSortSim | None = None,
        map_func: FIndexedSimMap | None = None) -> IndexedSimArray:
    sim_m = similarity.raw
    indices = np.arange(0, sim_m.shape[1], dtype=sim_m.dtype)
    active_sim: FloatArray = sim_m[sim_row]

    sort_similarity_arr = active_sim.T
    if pre_sort_sim is not None:
        sort_similarity_arr = pre_sort_sim(sort_similarity_arr)

    indexed_active_sim = np.column_stack((indices, active_sim.T))
    reversed_sorted_index = sort_similarity_arr.argsort()[::-1]
    desc_sim = indexed_active_sim[reversed_sorted_index]

    if map_func is not None:
        return map_func(desc_sim)
    return desc_sim


@cache
def indexed_support(sup_row: int, support: Support) -> IndexedSimArray:
    sup_m = support.raw
    indices = np.arange(0, sup_m.shape[1], dtype=sup_m.dtype)
    return np.column_stack((indices, sup_m[sup_row].T))
