from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from .predictors.item_based_cf import item_based_cf
from .predictors.slope_one_cf import slope_one_cf
from .presets import dynamic_presets, presets

if TYPE_CHECKING:
    from .config import Config
    from .core.train import CFPredictor
    from .typing import RatingMatrix


class Variant(TypedDict):
    predictor: CFPredictor
    conf: Config


# Got from experiments.
SIMPLE_COS_BEST_K = 10
SIMPLE_CORR_BEST_K = 11


def build_user_based_conf_list(r: RatingMatrix) -> dict[str, Config]:
    return {
    # Basic
        'cos':
            presets['cos'],
        'corr':
            presets['corr'],

    # Case Amplification - 2.5
        'cos_amp_2.5':
            presets['cos'] * dynamic_presets['case_amp'](2.5),
        'corr_amp_2.5':
            presets['corr'] * dynamic_presets['case_amp'](2.5),

    # Case Amplification - 1.5
        'cos_amp_1.5':
            presets['cos'] * dynamic_presets['case_amp'](1.5),
        'corr_amp_1.5':
            presets['corr'] * dynamic_presets['case_amp'](1.5),

    # Case Amplification - 3.5
        'cos_amp_3.5':
            presets['cos'] * dynamic_presets['case_amp'](3.5),
        'corr_amp_3.5':
            presets['corr'] * dynamic_presets['case_amp'](3.5),

    # IUF
        'cos_iuf':
            presets['cos'] + dynamic_presets['iuf'](r, False),
        'corr_iuf':
            presets['corr'] + dynamic_presets['iuf'](r, False),

    # Case Amplification - 2.5 + IUF
        'cos_iuf_amp_2.5':
            presets['cos'] * dynamic_presets['case_amp'](2.5) +
            dynamic_presets['iuf'](r, False),
        'corr_iuf_amp_2.5':
            presets['corr'] * dynamic_presets['case_amp'](2.5) +
            dynamic_presets['iuf'](r, False),
    }


vanilla_variants: list[Variant] = [
    # --- User-Based CF ---
    # User-Based CF variants are run with best_k_user_based

    # --- Item-Based CF ---
    {
        'predictor': item_based_cf,
        'conf': presets['cos'] + presets['item_based_k']
    },
    {
        'predictor': item_based_cf,
        'conf': presets['corr'] + presets['item_based_k']
    },
    {
        'predictor': item_based_cf,
        'conf': presets['adj_cos'] + presets['item_based_k']
    },

    # --- Slope-One CF ---
    {
        'predictor': slope_one_cf,
        'conf': presets['slope_one']
    }
]


def build_case_amp_variants(rho: float) -> list[Variant]:
    return [{
        'predictor': variant['predictor'],
        'conf': variant['conf'] * dynamic_presets['case_amp'](rho)
    } for variant in filter(lambda v: v['predictor'].__name__ != 'slope_one_cf',
                            vanilla_variants)]


default_case_amp_variants: list[Variant] = build_case_amp_variants(2.5)


def build_iuf_variants(r: RatingMatrix) -> list[Variant]:
    return [{
        'predictor':
            variant['predictor'],
        'conf':
            variant['conf'] + dynamic_presets['iuf'](
                r, item_based=variant['predictor'].__name__ != 'user_based_cf')
    } for variant in filter(lambda v: v['predictor'].__name__ != 'slope_one_cf',
                            vanilla_variants)]


def build_case_amp_iuf_variants(rho: float, r: RatingMatrix) -> list[Variant]:
    return [{
        'predictor':
            variant['predictor'],
        'conf':
            variant['conf'] * dynamic_presets['case_amp'](rho) +
            dynamic_presets['iuf'](
                r, item_based=variant['predictor'].__name__ != 'user_based_cf')
    } for variant in filter(lambda v: v['predictor'].__name__ != 'slope_one_cf',
                            vanilla_variants)]
