from typing import Any, Callable, TypedDict

import numpy as np

from .config import Config
from .core.knn.funcs.prediction_func import (
    adj_diff_weighted_average,
    diff_weighted_average,
    weighted_average,
)
from .core.knn.funcs.similarity_func import (
    adjusted_cosine_similarity,
    cosine_similarity,
    pearson_correlation,
)
from .core.shared.util_funcs import build_case_amplification, inverse_user_frequency

DynamicConfig = Callable[..., Config]


def item_based_dynamic_k(active_row):
    return np.ma.count(active_row)


class _Presets(TypedDict):
    corr: Config
    cos: Config
    case_amp: Config
    item_based: Config
    adj_cos: Config


class _DynamicPresets(TypedDict):
    case_amp: DynamicConfig
    iuf: DynamicConfig


presets: _Presets = {
    'corr':
        Config(sim_scheme=pearson_correlation,
               prediction=diff_weighted_average,
               pre_sort_sim=abs),    # type: ignore
    'cos':
        Config(sim_scheme=cosine_similarity, prediction=weighted_average),
    'case_amp':
        Config(indexed_sim_map=build_case_amplification(2.5)),
    'item_based_k':
        Config(knn_k=item_based_dynamic_k),
    'adj_cos':
        Config(sim_scheme=adjusted_cosine_similarity,
               prediction=adj_diff_weighted_average,
               pre_sort_sim=abs)    # type: ignore
}

dynamic_presets: _DynamicPresets = {
    'case_amp':
        lambda rho: Config(indexed_sim_map=build_case_amplification(rho)),
    'iuf':
        lambda r, item_based=False: Config(sim_weights=(inverse_user_frequency(
            r, item_based),) * 2)
}
