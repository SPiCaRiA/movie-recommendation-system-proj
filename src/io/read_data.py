from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .data_agg import aggregate_ratings

if TYPE_CHECKING:
    from ..typing import EntryArray, UserItemRatings

np.random.seed(0)

TRAIN_FNAME = 'data/train.txt'


def read_entries(fname: str) -> EntryArray:
    return pd.read_csv(fname, delimiter=' ', header=None) \
             .to_numpy(dtype=np.int32)


def read_split_entries(test_size: float) -> tuple[EntryArray, EntryArray]:
    '''
    Returns a tuple of entry arrays (train, test), derived from the raw training 
    data for cross validation.
    '''
    entry_arr = read_entries(TRAIN_FNAME)
    np.random.shuffle(entry_arr)

    if test_size < 1:
        bound = int(entry_arr.shape[0] * test_size)
    else:
        bound = int(test_size)

    train = entry_arr[bound:, ...]
    test = entry_arr[:bound, ...]

    return train, test


def readall_train() -> EntryArray:
    return read_entries(TRAIN_FNAME)
