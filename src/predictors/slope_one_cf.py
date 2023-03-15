from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.cf import indexed_support, support_matrix
from ..core.cf.similarity import similarity_matrix
from ..typing import float_array

if TYPE_CHECKING:
    from ..config import Config
    from ..typing import (
        PredictionArray,
        Questions,
        Similarity,
        Support,
        UserItemRatings,
    )


def slope_one_cf(ratings: UserItemRatings,
                 active_ratings: UserItemRatings,
                 questions: Questions,
                 config: Config,
                 item_diff: Similarity | None = None,
                 item_diff_sup: Support | None = None) -> PredictionArray:
    if item_diff is None:
        item_diff = similarity_matrix(ratings.raw.T,
                                      ratings.raw.T,
                                      config.sim_scheme,
                                      config.sim_fill_value,
                                      weights=config.sim_weights)

    if item_diff_sup is None:
        item_diff_sup = support_matrix(ratings.raw.T, ratings.raw.T)

    # In real world, active users do not come together at once. But for the sake
    # of this project, we batch and stack them with the training ratings.
    stacked_ratings = ratings + active_ratings
    predictions = float_array(questions.raw.shape[0])
    for i, (user_id, movie_id, _) in enumerate(questions.raw):
        user_ind = stacked_ratings.ind_user_id(user_id)
        movie_ind = stacked_ratings.ind_movie_id(movie_id)

        indexed_sup = indexed_support(movie_ind, item_diff_sup)

        predictions[i] = config.prediction(
            -1,    # Not a knn CF, use a negative value to escape k == 0 check
            user_ind,
            movie_ind,
            stacked_ratings[user_id] - item_diff.raw[:, movie_ind],
            stacked_ratings.raw.T,
            indexed_sup)

    return predictions
