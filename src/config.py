from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .typing import (
        FDynamicK,
        FIndexedSimMap,
        FloatMatrix,
        FPrediction,
        FPreSortSim,
        FSimilarity,
    )

ConfigFieldMissing = lambda name: ValueError(f'config field {name} missing.')


class Config():

    def __init__(
        self,
        knn_k: int | FDynamicK | None = None,
        sim_scheme: FSimilarity | None = None,
        prediction: FPrediction | None = None,
        sim_fill_value: int | None = None,
        indexed_sim_map: FIndexedSimMap | None = None,
        sim_weights: tuple[FloatMatrix, FloatMatrix] | None = None,
        pre_sort_sim: FPreSortSim | None = None,
    ):
        self._knn_k = knn_k
        self._sim_scheme = sim_scheme
        self._prediction = prediction
        self._sim_fill_value = sim_fill_value
        self._indexed_sim_map = indexed_sim_map
        self._sim_weights = sim_weights
        self._pre_sort_sim = pre_sort_sim

    def __add__(self, other: 'dict[str, Any] | Config') -> 'Config':
        '''
        Merge two configs.
        '''
        if isinstance(other, Config):
            config_dict = {}
            for field, value in self.__dict__.items():
                # Remove the '_' prefix of the private fields before sending
                # to the constructor.
                config_dict[field[1:]] = other[field] or value
            return Config(**config_dict)
        return self + Config(**other)

    def __mul__(self, other: 'dict[str, Any] | Config') -> 'Config':
        '''
        Merge two configs.
        ∀f ∈ {callable fields in self} and ∀g ∈ {callable fields in other}, make
        them g∘f in the resulting new config. Non-callable fields are treated
        the same as self & other.
        '''

        def compose(f, g):
            return lambda *args, **kwargs: g(f(*args, **kwargs))

        if isinstance(other, Config):
            res = self + other
            for field, value in self.__dict__.items():
                if callable(value) and callable(other[field]):
                    composed = compose(value, other[field])
                    composed.__name__ = other[
                        field].__name__ + '∘' + value.__name__
                    res[field] = composed
            return res
        return self * Config(**other)

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value

    def __str__(self) -> str:
        res = ''
        for k, v in self.__dict__.items():
            res += f'{k[1:]}: '
            # By type
            if isinstance(v, list):
                res += f'{[i.__name__ for i in v]}\n'
            elif isinstance(v, tuple):
                res += f'({str([type(i) for i in v])[1:-1]})\n'
            elif callable(v):
                res += f'{v.__name__ if v is not None else "None"}\n'

            # Special cases by name
            elif k == '_sim_fill_value':
                res += f'{v if v is not None else 0}\n'

            else:
                res += f'{v if v is not None else "None"}\n'
        return res

    @property
    def knn_k(self) -> int | FDynamicK:
        if self._knn_k is None:
            raise ConfigFieldMissing('knn_k')
        return self._knn_k

    @property
    def sim_scheme(self) -> FSimilarity:
        if self._sim_scheme is None:
            raise ConfigFieldMissing('sim_scheme')
        return self._sim_scheme

    @property
    def prediction(self) -> FPrediction:
        if self._prediction is None:
            raise ConfigFieldMissing('prediction')
        return self._prediction

    @property
    def sim_fill_value(self) -> int:
        # 0 is the default similarity fill value.
        return self._sim_fill_value if self._sim_fill_value is not None else 0

    @property
    def indexed_sim_map(self) -> FIndexedSimMap | None:
        return self._indexed_sim_map

    @property
    def sim_weights(self) -> tuple[FloatMatrix, FloatMatrix] | None:
        return self._sim_weights

    @property
    def pre_sort_sim(self) -> FPreSortSim | None:
        return self._pre_sort_sim
