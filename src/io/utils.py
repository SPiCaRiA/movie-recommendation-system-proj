from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ..typing import NpFloat, Similarity

if TYPE_CHECKING:
    from ..typing import SimilarityMatrix


def report_knn_test(method: str, k: int, mae: float):
    print(f'Running {method}-KNN on K={k}: {mae}')


def write_similarity(fname: str,
                     similarity: Similarity | SimilarityMatrix) -> None:
    if isinstance(similarity, Similarity):
        df = pd.DataFrame(similarity.raw)
    else:
        df = pd.DataFrame(similarity)
    df.to_csv(fname, index=False, header=False)


def read_similarity(fname: str) -> Similarity:
    matrix = pd.read_csv(fname, header=None).to_numpy(NpFloat)
    return Similarity(matrix)
