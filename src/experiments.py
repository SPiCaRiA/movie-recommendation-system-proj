from __future__ import annotations

from typing import TYPE_CHECKING

from .config import Config
from .core.cf import similarity_matrix
from .core.train import train_cf
from .io import (
    aggregate_all,
    aggregate_cross_validation,
    read_entries,
    read_matrix,
    read_split_entries,
    report_cf_test,
    write_matrix,
)
from .loss import loss_mae
from .predictors.item_based_cf import item_based_cf
from .predictors.slope_one_cf import slope_one_cf
from .predictors.user_based_cf import user_based_cf
from .presets import dynamic_presets, presets
from .typing import EntryArray, Similarity
from .utils import round_prediction
from .variants import (
    basic_variants,
    build_case_amp_iuf_variants,
    build_case_amp_variants,
    build_iuf_variants,
    default_case_amp_variants,
)

if TYPE_CHECKING:
    from .typing import Questions, UserItemRatings
    from .variants import Variant


def _run_variants(variants: list[Variant], r: UserItemRatings,
                  a: UserItemRatings, q: Questions):
    for variant in variants:
        predictor = variant['predictor']
        conf = variant['conf']
        predictions = predictor(r, a, q, conf)
        mae = loss_mae(q.ground_truth(),
                       [round_prediction(pred) for pred in predictions])
        report_cf_test(predictor.__name__, conf, mae)


class SimpleRun():
    train_arr, test_arr = read_split_entries(0.05)
    r, a, q = aggregate_cross_validation(train_arr, test_arr)

    @staticmethod
    def case_amplification_variants():
        # ρ = 2.5
        _run_variants(default_case_amp_variants, SimpleRun.r, SimpleRun.a,
                      SimpleRun.q)
        # ρ = 1.5
        _run_variants(build_case_amp_variants(1.5), SimpleRun.r, SimpleRun.a,
                      SimpleRun.q)
        # ρ = 3.5
        _run_variants(build_case_amp_variants(3.5), SimpleRun.r, SimpleRun.a,
                      SimpleRun.q)

    @staticmethod
    def iuf_variants():
        _run_variants(build_iuf_variants(SimpleRun.r.raw), SimpleRun.r,
                      SimpleRun.a, SimpleRun.q)

    @staticmethod
    def basic_variants():
        _run_variants(basic_variants, SimpleRun.r, SimpleRun.a, SimpleRun.q)

    @staticmethod
    def best_k_user_based():
        # Best k value for user-based CF
        SimpleRun._best_k_user_based(presets['cos'], list(range(1, 50)))
        SimpleRun._best_k_user_based(presets['corr'], list(range(1, 50)))

    @staticmethod
    def _best_k_user_based(conf: Config, k_range: list[int]):
        conf_list = [conf + {'knn_k': k} for k in k_range]
        best_mae, best_conf = train_cf(SimpleRun.r,
                                       SimpleRun.a,
                                       SimpleRun.q,
                                       user_based_cf,
                                       conf_list,
                                       verbosity=0)
        print(
            f'Best K value for {conf.sim_scheme.__name__} is {best_conf.knn_k}'
            + f', with MAE {best_mae}')


if __name__ == '__main__':
    SimpleRun.basic_variants()
