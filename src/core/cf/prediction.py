from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generator

from ...typing import IntMaskedArray

if TYPE_CHECKING:
    from ...typing import FPrediction, IndexedSimArray, RatingMatrix

Accessor = Callable[[], float]
PredictionGenerator = Generator[Accessor, tuple[float, int], None]


def prediction_func(
        build_generator: Callable[[int, int, RatingMatrix, IntMaskedArray],
                                  PredictionGenerator],
        filter_by_weight: Callable[[float], bool] | None = None) -> FPrediction:

    def wrapped(k: int, col: int, active_row: int, active_user: IntMaskedArray,
                r: RatingMatrix, indexed_desc_sim: IndexedSimArray) -> float:
        gen = build_generator(col, active_row, r, active_user)
        get_res = next(gen)

        for i_row, weight in indexed_desc_sim:
            if k == 0:
                break

            row = int(i_row)
            if r.mask[row, col]:
                # No value
                continue

            if (filter_by_weight(weight)
                    if filter_by_weight is not None else weight == 0):
                break

            k -= 1
            gen.send((weight, row))

        gen.close()
        return get_res()

    # Restore function name from the generator builder.
    wrapped.__name__ = build_generator.__name__

    return wrapped
