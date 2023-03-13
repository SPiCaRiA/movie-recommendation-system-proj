from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...typing import FIndexedSimMap, FloatMatrix, IndexedSimArray, RatingMatrix


def inverse_user_frequency(r: RatingMatrix,
                           item_based: bool = False) -> FloatMatrix:
    '''
    Returns an 1 x i vector of IUF weight for movies. To work with item_based
    CF, set the flag to True to get the i x 1 vector for broadcasting.
    '''
    m = len(r)
    m_j = np.apply_along_axis(lambda col: col.count() + 1, 0, r)
    weight_m = np.log(m / m_j)
    return weight_m[:, None] if item_based else weight_m


def build_case_amplification(rho: float) -> FIndexedSimMap:

    def case_amplification(index_sim_arr: IndexedSimArray) -> IndexedSimArray:
        sim_arr = index_sim_arr[:, 1]
        index_sim_arr[:, 1] = sim_arr * (abs(sim_arr)**(rho - 1))
        return index_sim_arr

    case_amplification.__name__ += f'(ρ={rho})'
    return case_amplification
