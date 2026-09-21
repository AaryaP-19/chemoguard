"""What units each estimator expects, declared rather than inferred.

CLAUDE.md Finding 7: the training script decided its unit convention by
inspecting the data in front of it (``if X['PLATELETS'].mean() > 1000: ... /
1000``) and then saved the estimator with a metadata file listing feature
*names* only. Whether the division happened was a property of the training
cohort, not of the column. Retrain on a neutropenic cohort -- the population
this tool is aimed at -- and the mean falls below the threshold, no division
occurs, the model learns its boundary at 1500 instead of 1.5, and nothing in
the saved artefact records which it was.

So the unit convention is part of the model contract and lives here, in
source, next to the feature order. Two rules follow:

* A new estimator ships with its contract updated in this file, in the same
  change. An estimator whose contract is not declared here cannot be served.
* ``features_for`` is the only place in the application where a Quantity
  becomes a bare float for a model. It converts through ``Quantity.to`` and
  hands the result straight to ``.predict()``.

The units declared below reproduce what the quarantined artefacts expected:
feature order from ``train_risk_prediction_models.py:53``, ``:76`` and
``:104``, and thousand/uL for the counts, which is where those three landed
(the anaemia model via the pre-divided CSV of Finding 3, the other two via
the ``.mean() > 1000`` check of Finding 7). They are written down here not
because thousand/uL is the right choice -- the NHANES replacements should
declare whatever they are actually trained on -- but because the convention
has to be recorded somewhere that a retrain cannot silently change.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from cbc import CBCPanel


class ContractError(RuntimeError):
    """An estimator and its declared contract disagree."""


@dataclass(frozen=True)
class Feature:
    """One column of a feature matrix: which analyte, in which unit."""

    analyte: str
    unit: str

    def as_dict(self) -> dict:
        return {"analyte": self.analyte, "unit": self.unit}


@dataclass(frozen=True)
class FeatureContract:
    """The ordered, unit-bearing input signature of one estimator."""

    model: str
    features: tuple

    @property
    def analytes(self) -> tuple:
        return tuple(f.analyte for f in self.features)

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "features": [f.as_dict() for f in self.features],
        }

    def __len__(self) -> int:
        return len(self.features)


def _contract(model: str, *pairs) -> FeatureContract:
    return FeatureContract(model, tuple(Feature(a, u) for a, u in pairs))


MODEL_CONTRACTS = {
    c.model: c
    for c in (
        _contract(
            "anemia",
            ("rbc", "million/uL"),
            ("wbc", "thousand/uL"),
            ("platelets", "thousand/uL"),
        ),
        _contract(
            "thrombocytopenia",
            ("rbc", "million/uL"),
            ("wbc", "thousand/uL"),
            ("platelets", "thousand/uL"),
        ),
        _contract(
            "neutropenia",
            ("rbc", "million/uL"),
            ("wbc", "thousand/uL"),
            ("platelets", "thousand/uL"),
            ("anc", "thousand/uL"),
        ),
    )
}


def contract_for(model: str) -> FeatureContract:
    try:
        return MODEL_CONTRACTS[model]
    except KeyError:
        raise ContractError(
            "No feature contract declared for model {0!r}. An estimator "
            "cannot be served until its units are declared in "
            "model_contract.MODEL_CONTRACTS.".format(model)
        ) from None


def features_for(model: str, panel: CBCPanel):
    """Build one feature row for an estimator from a panel of Quantities.

    Returns ``(row, record)`` -- a list of floats in the contract's order,
    and the per-feature record of what was fed in, for the prediction log.

    The single Quantity-to-float boundary in the application. Every float in
    the returned row was produced by ``Quantity.to`` against a unit named in
    this module; none was produced by inspecting a magnitude.

    Raises KeyError if the panel is missing an analyte the contract needs.
    """
    contract = contract_for(model)
    row, record = [], []

    for feature in contract.features:
        quantity = panel.get(feature.analyte)
        if quantity is None:
            raise KeyError(
                "{0} model needs {1}, which the panel does not carry.".format(
                    model, feature.analyte
                )
            )
        value = quantity.magnitude_in(feature.unit)
        row.append(value)
        record.append(
            {
                "analyte": feature.analyte,
                "fed_to_model": {"value": value, "unit": feature.unit},
                "canonical": quantity.as_dict(),
            }
        )

    return row, record


def check_estimator(model: str, estimator) -> None:
    """Refuse an estimator whose arity disagrees with its declared contract.

    A cheap guard, but it catches the case that matters: swapping in an
    artefact trained on a different feature set while the contract here still
    describes the old one. scikit-learn records ``n_features_in_`` on fit.
    """
    contract = contract_for(model)
    arity = getattr(estimator, "n_features_in_", None)
    if arity is not None and arity != len(contract):
        raise ContractError(
            "{0} estimator takes {1} features but the declared contract has "
            "{2} ({3}). Refusing to serve it.".format(
                model, arity, len(contract), ", ".join(contract.analytes)
            )
        )


def model_fingerprint(path: str):
    """A content hash of a model artefact, used as its version.

    The artefacts carry no version of their own -- ``save_models()`` writes
    feature names and nothing else (train_risk_prediction_models.py:293) --
    so the file's own digest is the only identifier that cannot drift from
    what was actually loaded. Returns None if the file is absent.
    """
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()[:16]
