from typing import Annotated, Callable, Literal

import numpy as np
import numpy.typing as npt

# Unify numpy scalar types used across this project.
NpFloat = np.float64
NpInt = np.int32

# Readable aliases of Literal for annotated types.
Shape = Literal
Represent = Literal

# Note: the annotated array types are nothing different from the original numpy
#       types. We just give additional information when creating them to hint
#       the usage (or characteristics) of the arrays.
#       Be aware that the PEP 593 explicitly says that type checkers should only
#       check base on the first item in an annotated type.

# --- Basic Numpy Array Aliases ---
_IntArray = npt.NDArray[NpInt]
_FloatArray = npt.NDArray[NpFloat]
_IntMaskedArray = Annotated[np.ma.MaskedArray[NpInt, np.dtype[np.int_]], NpInt]
_FloatMaskedArray = Annotated[np.ma.MaskedArray[NpInt, np.dtype[np.int_]],
                              NpFloat]

# --- Shape Related Array Aliases ---
# 1-d Arrays
IntArray = Annotated[_IntArray, Shape['_,']]
FloatArray = Annotated[_FloatArray, Shape['_,']]
IntMaskedArray = Annotated[_IntMaskedArray, Shape['_,']]
FloatMaskedArray = Annotated[_FloatMaskedArray, Shape['_,']]

# Matrices
IntMatrix = Annotated[_IntArray, Shape['_, _']]
FloatMatrix = Annotated[_FloatArray, Shape['_, _']]
IntMaskedMatrix = Annotated[_IntMaskedArray, Shape['_, _']]
FloatMaskedMatrix = Annotated[_FloatMaskedArray, Shape['_, _']]

# --- High-Level Representations ---
RatingMatrix = Annotated[IntMaskedMatrix, Represent['Rating']]
SimilarityMatrix = Annotated[FloatMatrix, Represent['Similarity']]
NanSimilarityMatrix = Annotated[FloatMaskedMatrix, Represent['NanSimilarity']]
EntryArray = Annotated[IntMatrix, Shape['_, 3'], Represent['Entry']]
IndexedSimArray = Annotated[FloatMatrix, Shape['_, 2'], Represent['IndexedSim']]
PredictionArray = Annotated[FloatArray, Represent['Prediction']]
SupportMatrix = Annotated[FloatMatrix, Represent['Support']]

# --- Function Types ---
FSimilarity = Callable[[RatingMatrix, RatingMatrix], NanSimilarityMatrix]
FPrediction = Callable[
    [int, int, int, IntMaskedArray, RatingMatrix, IndexedSimArray], float]
FPreSortSim = Callable[[FloatArray], FloatArray]
FIndexedSimMap = Callable[[IndexedSimArray], IndexedSimArray]
FDynamicK = Callable[[IntMaskedArray], int]
