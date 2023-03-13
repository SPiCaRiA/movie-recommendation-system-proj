from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..typing import Questions, UserItemRatings, int_masked_array, int_masked_matrix

if TYPE_CHECKING:
    from ..typing import EntryArray, IntMaskedArray


def aggregate_all(
    train_arr: EntryArray, active_arr: EntryArray
) -> tuple[UserItemRatings, UserItemRatings, Questions]:
    ratings = aggregate_ratings(train_arr)
    return ratings, aggregate_ratings(
        active_arr,
        (None, ratings.raw.shape[1])), aggregate_questions(active_arr)


def aggregate_ratings(
    entry_arr: EntryArray,
    shape: tuple[np.int_ | int | None, np.int_ | int | None] | None = None
) -> UserItemRatings:
    '''
    Aggregate rating entries into masked array matrix. Questions (i.e. entries
    with rating 0) will be automatically ignored.
    '''
    row_num = shape[0] if shape and shape[0] else len(
        np.unique(entry_arr[:, 0:1].flatten()))
    col_num = shape[1] if shape and shape[1] else np.max(
        entry_arr[:, 1:2].flatten())

    matrix = int_masked_matrix(True, (row_num, col_num))

    user_indices = {}
    index = 0
    for user_id, movie_id, rating in entry_arr:
        if rating != 0:
            if not user_id in user_indices:
                user_indices[user_id] = index
                index += 1
            matrix[user_indices[user_id], movie_id - 1] = rating

    return UserItemRatings(matrix, user_indices)


def aggregate_questions(entry_arr: EntryArray,
                        contains_ans: bool = False) -> Questions:
    '''
    Extract questions from an entry array that contains question entries. When 
    there is no corresponding ground truth, answer shall be 0.
    Returns (questions, active_users); active_users may be None if entry_arr
    only contains questions.

    If contains_ans is true, all entries are regarded as questions. Otherwise,
    only the entries with rating 0 are questions.
    '''
    return Questions(entry_arr, contains_ans)


def aggregate_cross_validation(
        train_arr: EntryArray, test_arr: EntryArray
) -> tuple[UserItemRatings, UserItemRatings, Questions]:
    max_train = np.max(train_arr[:, 1:2].flatten())
    max_test = np.max(test_arr[:, 1:2].flatten())
    item_num = np.max((max_train, max_test))

    ratings = aggregate_ratings(train_arr, shape=(None, item_num))

    return ratings, active_users_from_ratings(
        test_arr, ratings), aggregate_questions(test_arr, contains_ans=True)


# def active_users_from_entries(entry_arr: EntryArray,
#                               num_movies: int) -> UserItemRatings:
#     active_user_indices = {}
#     active_list: list[IntMaskedArray] = []
#     for user_id, movie_id, rating in entry_arr:
#         if not user_id in active_user_indices:
#             active_user_indices[user_id] = len(active_list)
#             active_list.append(int_masked_array(True, num_movies))

#         if rating != 0:
#             active_list[active_user_indices[user_id]][movie_id - 1] = rating

#     return UserItemRatings(np.ma.array(active_list), active_user_indices)


def active_users_from_ratings(entry_arr: EntryArray,
                              ratings: UserItemRatings) -> UserItemRatings:
    '''
    Build active user ratings from the existing ratings based on active user_ids.
    '''
    active_user_ids = np.unique(entry_arr[:, 0])

    active_user_indices = {}
    rows_to_take: list[int] = []
    for user_id in active_user_ids:
        index = ratings.ind_user_id(user_id)
        active_user_indices[user_id] = len(rows_to_take)
        rows_to_take.append(index)

    return UserItemRatings(ratings.raw[rows_to_take], active_user_indices)
