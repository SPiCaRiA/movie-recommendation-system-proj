from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ..typing import HashableMatrix, NpFloat

if TYPE_CHECKING:
    from ..typing import FloatMatrix


def report_knn_test(method: str, k: int, mae: float):
    print(f'Running {method}-KNN on K={k}: {mae}')


def write_matrix(fname: str, matrix: HashableMatrix | FloatMatrix) -> None:
    if isinstance(matrix, HashableMatrix):
        df = pd.DataFrame(matrix.raw)
    else:
        df = pd.DataFrame(matrix)
    df.to_csv(fname, index=False, header=False)


def read_matrix(fname: str) -> HashableMatrix:
    matrix = pd.read_csv(fname, header=None).to_numpy(NpFloat)
    return HashableMatrix(matrix)
