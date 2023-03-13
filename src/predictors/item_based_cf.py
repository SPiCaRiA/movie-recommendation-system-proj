from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.knn import indexed_desc_similarity, similarity_matrix
from ..typing import float_array

if TYPE_CHECKING:
    from ..config import Config
    from ..typing import PredictionArray, Questions, Similarity, UserItemRatings


def item_based_cf(ratings: UserItemRatings,
                  active_ratings: UserItemRatings,
                  questions: Questions,
                  config: Config,
                  item_similarity: Similarity | None = None) -> PredictionArray:
    stacked_ratings = ratings + active_ratings

    if item_similarity is None:
        item_similarity = similarity_matrix(stacked_ratings.raw.T,
                                            stacked_ratings.raw.T,
                                            config.sim_scheme,
                                            config.sim_fill_value,
                                            weights=config.sim_weights)

    predictions = float_array(questions.raw.shape[0])
    for i, (user_id, movie_id, _) in enumerate(questions.raw):
        movie_ind = stacked_ratings.ind_movie_id(movie_id)
        user_ind = stacked_ratings.ind_user_id(user_id)

        item_desc_sim = indexed_desc_similarity(movie_ind,
                                                item_similarity,
                                                config.pre_sort_sim,
                                                map_func=config.indexed_sim_map)

        active_user = stacked_ratings[user_id]
        if callable(config.knn_k):
            k = config.knn_k(active_user)
        else:
            k = config.knn_k

        predictions[i] = config.prediction(k, user_ind, movie_ind,
                                           stacked_ratings.raw[user_ind],
                                           stacked_ratings.raw.T, item_desc_sim)

    return predictions
