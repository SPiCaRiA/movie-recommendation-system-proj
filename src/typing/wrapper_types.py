from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, overload

import numpy as np

from .nparray_builder import int_array

if TYPE_CHECKING:
    from .type_aliases import (
        EntryArray,
        FloatArray,
        IntArray,
        IntMaskedArray,
        RatingMatrix,
        SimilarityMatrix,
    )


def IDIndexError(name: str, val: str = '', suffix: str = '') -> IndexError:
    return IndexError(f'{name}{val} out of range. {suffix}')


class UserItemRatings():

    def __init__(self, raw_matrix: RatingMatrix, user_indices: dict[int, int]):
        self.raw = raw_matrix
        self.raw.setflags(write=False)
        self._user_indices = user_indices

    @overload
    def ind_user_id(self, user_id: int) -> int:
        ...

    @overload
    def ind_user_id(self, user_id: list[int] | IntArray) -> IntArray:
        ...

    def ind_user_id(self,
                    user_id: int | list[int] | IntArray) -> int | IntArray:
        if isinstance(user_id, int) or isinstance(user_id, np.integer):
            if not user_id in self._user_indices:
                raise IDIndexError('user_id ', f'{user_id}',
                                   f'{self.raw.shape}')
            return self._user_indices[user_id]
        else:
            try:
                res = int_array([self._user_indices[i] for i in user_id])
            except KeyError as exc:
                raise IDIndexError('existing user_id') from exc
            return res

    @overload
    def ind_movie_id(self, movie_id: int) -> int:
        ...

    @overload
    def ind_movie_id(self, movie_id: list[int] | IntArray) -> IntArray:
        ...

    def ind_movie_id(self,
                     movie_id: int | list[int] | IntArray) -> int | IntArray:
        if isinstance(movie_id, int) or isinstance(movie_id, np.integer):
            if movie_id <= 0 or movie_id > self.raw.shape[1]:
                raise IDIndexError('movie_id ', f'{movie_id}',
                                   f'{self.raw.shape}')
            return movie_id - 1
        elif isinstance(movie_id, list):
            return int_array([i - 1 for i in movie_id])
        else:
            res = movie_id - 1
            if np.any(res, where=(res < 0) | (res >= self.raw.shape[1])):
                raise IDIndexError('existing movie_ids', '',
                                   f'{self.raw.shape}')
            return res

    def __getitem__(self, key: int | tuple[int | slice, int | slice]):
        '''
        Access user-item rating matrix with user_ids and movie_ids instead of 
        row and col indices.
        '''
        if isinstance(key, int) or isinstance(key, np.integer):
            return self.raw[self.ind_user_id(key)]
        else:
            row, col = key
            if isinstance(key, int) or isinstance(key, np.integer):
                row = self.ind_user_id(row)
            if isinstance(key, int) or isinstance(key, np.integer):
                col = self.ind_movie_id(col)
            return self.raw[row, col]

    def __add__(self, other: UserItemRatings) -> UserItemRatings:
        new_user_indices = {}
        new_user_indices.update(self._user_indices)

        # If any new users in other, append them to self.raw and update the
        # user_indices dict.
        offset = len(self.raw)
        rows_to_add: list[int] = []
        for user_id in other._user_indices:
            if not user_id in new_user_indices:
                new_user_indices[user_id] = len(rows_to_add) + offset
                rows_to_add.append(other.ind_user_id(user_id))

        return UserItemRatings(np.ma.vstack((self.raw, other.raw[rows_to_add])),
                               new_user_indices)


class Questions():

    def __init__(self, entry_arr: EntryArray, contains_ans: bool):
        '''
        Extract questions from an entry array that contains question entries. 
        When there is no corresponding ground truth, answer shall be 0.

        If contains_ans is true, all entries are regarded as questions. Otherwise,
        only the entries with rating 0 are questions.
        '''
        self.raw: EntryArray
        self._answers: None | list[int] | IntArray
        self._contains_ans = contains_ans

        if contains_ans:
            self.raw = entry_arr
            self._answers = entry_arr[:, 2]
        else:
            self.raw = entry_arr[entry_arr[:, 2] == 0]
            self._answers = None

        self.raw.setflags(write=False)

    @cached_property
    def questions(self) -> dict[int, list[int]]:
        '''
        Returns a dictionary of user ids and the list of missing movie ids.
        '''
        question_dict: dict[int, list[int]] = {}
        for user_id, movie_id, _ in self.raw:
            if not user_id in question_dict:
                question_dict[user_id] = []
            question_dict[user_id].append(movie_id)

        return question_dict

    def __getitem__(self, user_id: int) -> list[int]:
        '''
        Return an array of missing movie ids under the given user id.
        '''
        if not user_id in self.questions:
            raise IndexError(f'user_id {user_id} not found.')
        return self.questions[user_id]

    def _filter_bad_values(
        self, answers: list[int] | list[float] | IntArray | FloatArray
    ) -> list[tuple[int, int, int]]:
        res = []
        for i, ans in enumerate(answers):
            if ans < 1 or ans > 5:
                res.append((self.raw[i][0], self.raw[i][1], ans))
        return res

    def take_answers(self,
                     answers: list[int] | list[float] | IntArray | FloatArray,
                     force_update: bool = False,
                     validate: bool = True) -> 'Questions':
        if not force_update and self._contains_ans:
            raise ValueError('answers already exist.')
        self._contains_ans = True

        if (isinstance(answers, list) and isinstance(answers[0], float)) or \
            (isinstance(answers, np.ndarray) and np.issubdtype(answers.dtype, np.floating)):
            answers = [
            # Force rounding 0.x to 1
                1 if round(ans) == 0 and ans != 0 else round(ans)
                for ans in answers
            ]

        if validate and not len(
                bad_values := self._filter_bad_values(answers)) == 0:
            raise ValueError(f'invalid answer values {bad_values}.')

        self._answers = answers if isinstance(    # type: ignore
            answers, list) else answers.tolist()
        return self

    def __str__(self) -> str:
        res = ''
        for i, (user_id, movie_id, rating) in enumerate(self.raw):
            ans = self._answers[i] if self._answers is not None else rating
            res += f'{user_id} {movie_id} {ans}\n'
        return res


class Similarity():
    '''
    Hashable wrapper of SimilarityMatrix.
    '''

    def __init__(self, sim_matrix: SimilarityMatrix) -> None:
        self.raw = sim_matrix
        self.raw.setflags(write=False)

    @cached_property
    def _hash(self) -> int:
        '''
        Hash the similarity matrix with str conversion.
        We don't need a perfect hash for similarity matrices, since they should
        be very different.
        '''
        return hash(str(self.raw))

    def __hash__(self) -> int:
        return self._hash
