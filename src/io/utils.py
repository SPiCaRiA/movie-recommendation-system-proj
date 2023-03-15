from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ..typing import HashableMatrix, NpFloat

if TYPE_CHECKING:
    from ..config import Config
    from ..typing import FloatMatrix


def report_cf_test(method: str, conf: Config, mae: float):
    print(f'--------\tMAE: {mae}\t--------')
    print(f'Running {method}-KNN on:')
    print(conf)


def write_matrix(fname: str, matrix: HashableMatrix | FloatMatrix) -> None:
    if isinstance(matrix, HashableMatrix):
        df = pd.DataFrame(matrix.raw)
    else:
        df = pd.DataFrame(matrix)
    df.to_csv(fname, index=False, header=False)


def read_matrix(fname: str) -> HashableMatrix:
    matrix = pd.read_csv(fname, header=None).to_numpy(NpFloat)
    return HashableMatrix(matrix)
