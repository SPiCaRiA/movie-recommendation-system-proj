from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.knn import indexed_desc_similarity, similarity_matrix
from ..typing import float_array

if TYPE_CHECKING:
    from ..config import Config
    from ..typing import PredictionArray, Questions, Similarity, UserItemRatings


def user_based_cf(ratings: UserItemRatings,
                  active_ratings: UserItemRatings,
                  questions: Questions,
                  config: Config,
                  user_similarity: Similarity | None = None) -> PredictionArray:
    if user_similarity is None:
        user_similarity = similarity_matrix(active_ratings.raw,
                                            ratings.raw,
                                            config.sim_scheme,
                                            config.sim_fill_value,
                                            weights=config.sim_weights)

    predictions = float_array(questions.raw.shape[0])
    for i, (user_id, movie_id, _) in enumerate(questions.raw):
        movie_ind = active_ratings.ind_movie_id(movie_id)
        user_ind = active_ratings.ind_user_id(user_id)

        user_desc_sim = indexed_desc_similarity(user_ind, user_similarity,
                                                config.pre_sort_sim)

        active_user = active_ratings[user_id]
        if callable(config.knn_k):
            k = config.knn_k(active_user)
        else:
            k = config.knn_k

        predictions[i] = config.prediction(k, movie_ind, user_ind,
                                           active_ratings.raw[user_ind],
                                           ratings.raw, user_desc_sim)

    return predictions
