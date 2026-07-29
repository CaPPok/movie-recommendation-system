"""Version-independent persistence for the fitted TF-IDF vectorizer.

`joblib.dump(TfidfVectorizer)` writes a pickle of a scikit-learn estimator, and
that pickle only reloads correctly under a compatible scikit-learn. Measured on
2026-07-29: an artifact written by 1.8.0 loads under 1.2.2 without raising, then
fails at the first `transform()` with `NotFittedError: idf vector is not fitted`.
The newest scikit-learn inference image AWS publishes is 1.2, and the retraining
Processing Job runs on 1.4, so the three environments this project uses could
never agree on one pickle.

A fitted TF-IDF vectorizer is not complex state. It is a vocabulary, one idf
weight per vocabulary entry, and the analyzer settings that turn text into
tokens. Stored as JSON and `.npy`, all three outlive any library version, and
`transform` is reproduced from them with `CountVectorizer` -- whose fixed-
vocabulary `transform` is stable across versions -- plus arithmetic.

Verified bit-identical, not approximately: for the same inputs, the arrays this
module produces under scikit-learn 1.2.2 and the arrays the original 1.8.0
pickle produces differ by 0.0 exactly. Preserving the stored `dtype` is what
makes that exact rather than close; the vectorizer is fitted with
`dtype=np.float32`, and rebuilding in float64 leaves a ~2.5e-08 gap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

VOCABULARY_FILE = "vectorizer_vocabulary.json"
IDF_FILE = "vectorizer_idf.npy"
PARAMETERS_FILE = "vectorizer_params.json"
PORTABLE_FILES = (VOCABULARY_FILE, IDF_FILE, PARAMETERS_FILE)

# Everything `transform` consults. `max_features`, `min_df` and `max_df` shape
# which terms end up in the vocabulary during fit and have no effect afterwards,
# so they are deliberately absent.
_TRANSFORM_PARAMETERS = (
    "ngram_range",
    "lowercase",
    "token_pattern",
    "stop_words",
    "strip_accents",
    "analyzer",
    "binary",
    "sublinear_tf",
    "norm",
)


class PortableTfidfVectorizer:
    """Reproduces a fitted `TfidfVectorizer.transform` from portable state.

    Exposes `transform` and `vocabulary_` so callers can treat it as the
    estimator it replaces.
    """

    def __init__(
        self,
        vocabulary: dict[str, int],
        idf: np.ndarray,
        parameters: dict[str, Any],
    ) -> None:
        self.vocabulary_ = vocabulary
        self.parameters = parameters
        self.dtype = np.dtype(parameters.get("dtype", "float32"))
        self.idf_ = np.asarray(idf, dtype=self.dtype)
        self._counter = CountVectorizer(
            vocabulary=vocabulary,
            ngram_range=tuple(parameters["ngram_range"]),
            lowercase=parameters["lowercase"],
            token_pattern=parameters["token_pattern"],
            stop_words=parameters["stop_words"],
            strip_accents=parameters["strip_accents"],
            analyzer=parameters["analyzer"],
            binary=parameters["binary"],
        )
        self._idf_diagonal = sparse.diags(self.idf_)

    def transform(self, raw_documents: Iterable[str]) -> sparse.csr_matrix:
        matrix = self._counter.transform(raw_documents).astype(self.dtype)
        if self.parameters["sublinear_tf"]:
            # Applied to `.data` only: log is defined on stored counts, and every
            # absent term must stay structurally zero rather than become 1+log(0).
            matrix.data = 1.0 + np.log(matrix.data)
        matrix = (matrix @ self._idf_diagonal).tocsr()
        return normalize(matrix, norm=self.parameters["norm"], copy=False)


def extract_parameters(vectorizer: Any) -> dict[str, Any]:
    """Read the transform-relevant settings off a fitted `TfidfVectorizer`."""
    parameters: dict[str, Any] = {
        name: getattr(vectorizer, name) for name in _TRANSFORM_PARAMETERS
    }
    parameters["ngram_range"] = list(parameters["ngram_range"])
    parameters["lowercase"] = bool(parameters["lowercase"])
    parameters["binary"] = bool(parameters["binary"])
    parameters["sublinear_tf"] = bool(parameters["sublinear_tf"])
    parameters["dtype"] = np.dtype(vectorizer.dtype).name
    return parameters


def save_portable_vectorizer(vectorizer: Any, artifacts_dir: Path) -> dict[str, Any]:
    """Write the three portable files next to the existing artifacts."""
    artifacts_dir = Path(artifacts_dir)
    parameters = extract_parameters(vectorizer)
    vocabulary = {
        str(term): int(index) for term, index in vectorizer.vocabulary_.items()
    }
    idf = np.asarray(vectorizer.idf_, dtype=np.dtype(parameters["dtype"]))

    # Written through temporary names for the same reason the rest of this
    # package does: a partial artifact must never look like a complete one.
    for filename, payload in (
        (VOCABULARY_FILE, json.dumps(vocabulary, ensure_ascii=False)),
        (PARAMETERS_FILE, json.dumps(parameters, ensure_ascii=False, indent=2)),
    ):
        target = artifacts_dir / filename
        temporary = target.with_name(f"{target.name}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(target)

    idf_target = artifacts_dir / IDF_FILE
    idf_temporary = idf_target.with_name(f"{idf_target.name}.tmp.npy")
    np.save(idf_temporary, idf)
    idf_temporary.replace(idf_target)

    return {"terms": len(vocabulary), "idf": int(idf.shape[0]), "dtype": parameters["dtype"]}


def load_portable_vectorizer(artifacts_dir: Path) -> PortableTfidfVectorizer:
    """Rebuild the vectorizer from the portable files."""
    artifacts_dir = Path(artifacts_dir)
    missing = [name for name in PORTABLE_FILES if not (artifacts_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Thiếu artifact vectorizer dạng phổ thông: "
            + ", ".join(missing)
            + ". Chạy: python scripts/convert_vectorizer.py"
        )
    vocabulary = json.loads(
        (artifacts_dir / VOCABULARY_FILE).read_text(encoding="utf-8")
    )
    parameters = json.loads(
        (artifacts_dir / PARAMETERS_FILE).read_text(encoding="utf-8")
    )
    idf = np.load(artifacts_dir / IDF_FILE)
    return PortableTfidfVectorizer(vocabulary, idf, parameters)
