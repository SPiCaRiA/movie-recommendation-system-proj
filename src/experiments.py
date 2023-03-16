from __future__ import annotations

from typing import TYPE_CHECKING

from .core.train import train_cf
from .io import (
    aggregate_all,
    aggregate_cross_validation,
    read_entries,
    read_split_entries,
    readall_train,
    report_cf_test,
)
from .loss import loss_mae
from .predictors.user_based_cf import user_based_cf
from .presets import dynamic_presets, presets
from .utils import round_prediction
from .variants import (
    build_case_amp_iuf_variants,
    build_case_amp_variants,
    build_iuf_variants,
    build_user_based_conf_list,
    default_case_amp_variants,
    vanilla_variants,
)

if TYPE_CHECKING:
    from .config import Config
    from .typing import EntryArray, Questions, UserItemRatings
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


class CrossValidationExperiment():

    def __init__(self, train_arr: EntryArray, test_arr: EntryArray):
        self.raq = aggregate_cross_validation(train_arr, test_arr)
        self.r, self.a, self.q = self.raq

    def case_amplification_variants(self):
        # ρ = 2.5
        _run_variants(default_case_amp_variants, *self.raq)
        # ρ = 1.5
        _run_variants(build_case_amp_variants(1.5), *self.raq)
        # ρ = 3.5
        _run_variants(build_case_amp_variants(3.5), *self.raq)

    def iuf_variants(self):
        _run_variants(build_iuf_variants(self.r.raw), *self.raq)

    def case_amp_iuf_variants(self):
        _run_variants(build_case_amp_iuf_variants(2.5, self.r.raw), *self.raq)

    def vanilla_variants(self):
        _run_variants(vanilla_variants, *self.raq)

    def best_k_user_based(self):
        # Best k value for user-based CF
        for name, conf in build_user_based_conf_list(self.r.raw).items():
            self._best_k_user_based(conf, list(range(1, 50)), name)

    def _best_k_user_based(self, conf: Config, k_range: list[int], name: str):
        conf_list = [conf + {'knn_k': k} for k in k_range]
        best_mae, best_conf = train_cf(*self.raq,
                                       user_based_cf,
                                       conf_list,
                                       verbosity=0)
        print(f'Best K value for {name} is {best_conf.knn_k}' +
              f', with MAE {best_mae}')


simple_run = CrossValidationExperiment(*read_split_entries(0.05))
uvi_5 = CrossValidationExperiment(read_entries('data/uvi/train.uvi5.txt'),
                                  read_entries('data/uvi/test.uvi5.txt'))
uvi_20 = CrossValidationExperiment(read_entries('data/uvi/train.uvi20.txt'),
                                   read_entries('data/uvi/test.uvi20.txt'))


def predict_write_real():
    fnames = ['test5', 'test10', 'test20']
    train_arr = readall_train()

    # --- Change here ---
    predictor = user_based_cf
    infix = 'item_corr'
    # --- Change here ---

    for fname in fnames:
        # --- Change here ---
        conf = presets['corr'] + presets['item_based_k']
        # --- Change here ---
        test_arr = read_entries(f'data/task/{fname}.txt')
        r, a, q = aggregate_all(train_arr, test_arr)
        predictions = predictor(r, a, q, conf)
        q.take_answers(predictions, validate=False)

        with open(f'data/outputs/{fname}.{infix}.txt', 'w') as f:
            f.write(str(q))


if __name__ == '__main__':
    predict_write_real()
