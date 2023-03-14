from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.cf import indexed_support, support_matrix
from ..typing import float_array

if TYPE_CHECKING:
    from ..config import Config
    from ..typing import PredictionArray, Questions, Similarity, UserItemRatings


def slope_one(ratings: UserItemRatings,
              active_ratings: UserItemRatings,
              questions: Questions,
              config: Config,
              item_diff: Similarity | None = None) -> PredictionArray:
    stacked_ratings = ratings + active_ratings

    if item_diff is None:
        item_diff_matrix = config.sim_scheme(stacked_ratings.raw.T,
                                             stacked_ratings.raw.T).filled(
                                                 config.sim_fill_value)
    else:
        item_diff_matrix = item_diff.raw

    sup_m = support_matrix(stacked_ratings.raw.T, stacked_ratings.raw.T)

    predictions = float_array(questions.raw.shape[0])
    for i, (user_id, movie_id, _) in enumerate(questions.raw):
        user_ind = stacked_ratings.ind_user_id(user_id)
        movie_ind = stacked_ratings.ind_movie_id(movie_id)

        indexed_sup = indexed_support(movie_ind, sup_m)

        predictions[i] = config.prediction(
            -1,    # Not a knn CF, use a negative value to escape k == 0 check
            user_ind,
            movie_ind,
            stacked_ratings[user_id] - item_diff_matrix[:, movie_ind],
            stacked_ratings.raw.T,
            indexed_sup)

    return predictions
