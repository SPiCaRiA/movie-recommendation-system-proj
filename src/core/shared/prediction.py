from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generator

from ...typing import IntMaskedArray

if TYPE_CHECKING:
    from ...typing import FPrediction, IndexedSimArray, RatingMatrix

Accessor = Callable[[], float]
PredictionGenerator = Generator[Accessor, tuple[float, IntMaskedArray], None]


def prediction_func(
        build_generator: Callable[[int, IntMaskedArray, IntMaskedArray],
                                  PredictionGenerator],
        filter_by_weight: Callable[[float], bool] | None = None) -> FPrediction:

    def wrapped(k: int, col: int, active_row: int, active_user: IntMaskedArray,
                r: RatingMatrix, indexed_desc_sim: IndexedSimArray) -> float:
        gen = build_generator(col, r[active_row], active_user)
        get_res = next(gen)

        for i_row, weight in indexed_desc_sim:
            if k == 0:
                break

            row = int(i_row)
            if r.mask[row, col]:
                # No value
                continue

            if (filter_by_weight(weight) if filter_by_weight else weight == 0):
                break

            k -= 1
            gen.send((weight, r[row]))

        gen.close()
        return get_res()

    # Restore function name from its builder.
    wrapped.__name__ = build_generator.__name__

    return wrapped
