from __future__ import annotations

from typing import TYPE_CHECKING

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from .core.cf import similarity_matrix, support_matrix
from .io import aggregate_cross_validation, read_entries, read_split_entries
from .loss import loss_rmse
from .predictors import item_based_cf, slope_one_cf, user_based_cf
from .presets import dynamic_presets, presets

if TYPE_CHECKING:
    from .typing import Questions, UserItemRatings

search_space = [
    Real(0, 1, name='weight_slope_one'),
    Real(0, 1, name='weight_item_corr'),
    Real(0, 1, name='weight_user_corr_iuf'),
    Real(0, 1, name='weight_user_cos'),
]

# Pre-computed values.
# _r, _a, _q = aggregate_cross_validation(*read_split_entries(0.1))
_r, _a, _q = aggregate_cross_validation(
    read_entries('data/uvi/train.uvi5.extend.txt'),
    read_entries('data/uvi/test.uvi5.extend.txt'))

_slope_one_diff = similarity_matrix(_r.raw.T, _r.raw.T,
                                    presets['slope_one'].sim_scheme, 0)
_slope_one_sup = support_matrix(_r.raw.T, _r.raw.T)
_item_corr_sim = similarity_matrix(_r.raw.T, _r.raw.T,
                                   presets['corr'].sim_scheme, 0)
_user_corr_iuf_sim = similarity_matrix(_a.raw,
                                       _r.raw,
                                       presets['corr'].sim_scheme,
                                       0,
                                       weights=dynamic_presets['iuf'](
                                           _r.raw).sim_weights)
_user_cos_sim = similarity_matrix(_a.raw, _r.raw, presets['cos'].sim_scheme, 0)


def linear_ensembler(weight_slope_one: float = 1,
                     weight_item_corr: float = 1,
                     weight_user_corr_iuf: float = 1,
                     weight_user_cos: float = 1):
    slope_one_pred = slope_one_cf(_r,
                                  _a,
                                  _q,
                                  presets['slope_one'] +
                                  presets['item_based_k'],
                                  item_diff=_slope_one_diff,
                                  item_diff_sup=_slope_one_sup)
    item_corr_pred = item_based_cf(_r,
                                   _a,
                                   _q,
                                   presets['corr'] + presets['item_based_k'],
                                   item_similarity=_item_corr_sim)
    user_corr_iuf_pred = user_based_cf(_r,
                                       _a,
                                       _q,
                                       presets['corr'] + {'knn_k': 20},
                                       user_similarity=_user_corr_iuf_sim)
    user_cos_pred = user_based_cf(_r,
                                  _a,
                                  _q,
                                  presets['cos'] + {'knn_k': 10},
                                  user_similarity=_user_cos_sim)

    predictions = weight_slope_one * slope_one_pred + \
                weight_item_corr * item_corr_pred + \
                weight_user_corr_iuf * user_corr_iuf_pred + \
                weight_user_cos * user_cos_pred

    return predictions


@use_named_args(search_space)
def evaluate_ensembler(**hparams):
    predictions = linear_ensembler(**hparams)
    loss = loss_rmse(_q.ground_truth(), predictions)
    return loss


if __name__ == '__main__':
    result = gp_minimize(evaluate_ensembler, search_space)
    print(f'Best RMSE: {result.fun}')
    print(
        f'Best Parameters: w1={result.x[0]}, w2={result.x[1]}, w3={result.x[2]}, w4={result.x[3]}'
    )
