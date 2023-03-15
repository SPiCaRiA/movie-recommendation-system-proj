from __future__ import annotations

from typing import Callable, Concatenate, ParamSpec

from ..config import Config
from ..core.cf import similarity_matrix
from ..io import report_cf_test
from ..loss import loss_mae
from ..typing import PredictionArray, Questions, UserItemRatings

P = ParamSpec('P')
# CFPredictor = Callable[
#     [UserItemRatings, UserItemRatings, Questions, Config, Similarity | None],
#     PredictionArray]
CFPredictor = Callable[Concatenate[UserItemRatings, UserItemRatings, Questions,
                                   Config, P], PredictionArray]


def train_cf(ratings: UserItemRatings, active_ratings: UserItemRatings,
             questions: Questions, predictor: CFPredictor,
             conf_list: list[Config]) -> None:
    init_conf = conf_list[0]
    if predictor.__name__ == 'slope_one_cf':
        sim_m = init_conf.sim_scheme(ratings.raw.T, ratings.raw.T).filled(
            init_conf.sim_fill_value)
    elif predictor.__name__ == 'user_based_cf':
        sim_m = similarity_matrix(active_ratings.raw,
                                  ratings.raw,
                                  init_conf.sim_scheme,
                                  init_conf.sim_fill_value,
                                  weights=init_conf.sim_weights)
    elif predictor.__name__ == 'item_based_cf':
        sim_m = similarity_matrix(ratings.raw.T,
                                  ratings.raw.T,
                                  init_conf.sim_scheme,
                                  init_conf.sim_fill_value,
                                  weights=init_conf.sim_weights)
    else:
        raise NameError(f'predictor not found ({predictor.__name__})')

    best: tuple[float, Config | None] = (5, None)
    for conf in conf_list:
        # We can only train K value when it's not dynamic.
        assert isinstance(conf.knn_k, int)

        predictions = predictor(ratings, active_ratings, questions, conf, sim_m)
        # For training data, q should have ground truth as the 3rd column of the
        # entry array.
        mae = loss_mae(questions.raw[:, 2],
                       [round(pred) for pred in predictions])
        if mae < best[0]:
            best = (mae, conf)
        report_cf_test(predictor.__name__, conf, mae)

    print(f'\nBest: MAE {best[0]}\n{best[1]}')
