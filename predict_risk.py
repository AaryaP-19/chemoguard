"""
Command-line risk prediction against the trained models.

Every value must be given with its unit. There is no default and no
inference: the version of this script this replaces printed
``'cells/uL' if args.wbc > 1000 else 'thousands/uL'``, displaying a guess to
the operator as though it were a fact read off the report.

Usage:
    python predict_risk.py --rbc 4.5 --rbc-unit million/uL \
                           --wbc 8000 --wbc-unit cells/uL \
                           --platelets 250000 --platelets-unit cells/uL \
                           --anc 5000 --anc-unit cells/uL
"""

import argparse
import json
import os

import joblib
import numpy as np

import prediction_log
from cbc import ANALYTES, ImplausibleValueError, build_panel
from clinical_flags import flags_for
from model_contract import (
    ContractError,
    check_estimator,
    contract_for,
    features_for,
    model_fingerprint,
)
from units import UnitError, known_unit_symbols

MODEL_TYPES = ('anemia', 'neutropenia', 'thrombocytopenia')

#: Analytes this CLI accepts, each as a --<name> / --<name>-unit pair.
CLI_ANALYTES = ('rbc', 'wbc', 'platelets', 'anc', 'neutrophils_percent')


def load_models():
    """Load all trained models, checking each against its declared contract."""
    models = {}

    for risk_type in MODEL_TYPES:
        model_path = f'models/{risk_type}_risk_model_latest.pkl'
        encoder_path = f'models/{risk_type}_risk_encoder_latest.pkl'
        scaler_path = f'models/{risk_type}_risk_scaler_latest.pkl'

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}. Please train models first.")

        estimator = joblib.load(model_path)
        # The arity guard from model_contract. An estimator whose feature
        # count disagrees with the units declared for it must not predict.
        check_estimator(risk_type, estimator)

        models[risk_type] = {
            'model': estimator,
            'encoder': joblib.load(encoder_path),
            'scaler': joblib.load(scaler_path) if os.path.exists(scaler_path) else None,
            'version': model_fingerprint(model_path),
        }

    return models


def predict_risk(models, panel, request_id=None):
    """Predict all three risks from a panel of Quantities.

    Takes a CBCPanel. There is no signature here that accepts a bare float,
    which is the point: a caller cannot reach the estimators without having
    said what its numbers mean.
    """
    panel = panel.with_derived_anc()
    request_id = request_id or prediction_log.new_prediction_id()

    predictions = {}
    feature_records = {}
    versions = {}

    for risk_type in MODEL_TYPES:
        if risk_type == 'neutropenia' and panel.get('anc') is None:
            predictions[risk_type] = {
                'risk': None,
                'probabilities': None,
                'error': 'ANC or a neutrophil percentage is required for neutropenia.',
            }
            continue

        row, record = features_for(risk_type, panel)
        features = np.array([row])

        estimator = models[risk_type]['model']
        encoder = models[risk_type]['encoder']
        predicted = estimator.predict(features)[0]

        probabilities = None
        if hasattr(estimator, 'predict_proba'):
            proba = estimator.predict_proba(features)[0]
            probabilities = {str(k): float(v) for k, v in zip(encoder.classes_, proba)}

        predictions[risk_type] = {
            'risk': str(encoder.inverse_transform([predicted])[0]),
            'probabilities': probabilities,
            'model_version': models[risk_type]['version'],
            'feature_units': contract_for(risk_type).as_dict()['features'],
        }
        feature_records[risk_type] = record
        versions[risk_type] = models[risk_type]['version']

    flags = flags_for(panel)
    prediction_log.log_prediction(
        request_id=request_id,
        panel=panel,
        predictions=predictions,
        feature_records=feature_records,
        model_versions=versions,
        flags=flags,
    )
    return predictions, flags


def build_parser():
    parser = argparse.ArgumentParser(
        description='Predict haematological risks using trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Every value needs its unit. Both of these are valid and they mean the same
thing; neither is inferred from the size of the number:

  python predict_risk.py --rbc 4.5 --rbc-unit million/uL \\
      --wbc 8 --wbc-unit thousand/uL \\
      --platelets 250 --platelets-unit thousand/uL

  python predict_risk.py --rbc 4.5 --rbc-unit million/uL \\
      --wbc 8000 --wbc-unit cells/uL \\
      --platelets 250000 --platelets-unit cells/uL

Known units: {0}
        """.format(", ".join(known_unit_symbols())),
    )

    for key in CLI_ANALYTES:
        analyte = ANALYTES[key]
        flag = '--' + key.replace('_', '-')
        required = key in ('rbc', 'wbc', 'platelets')
        parser.add_argument(
            flag, dest=key, type=float, required=required,
            help='{0} value.'.format(analyte.label),
        )
        parser.add_argument(
            flag + '-unit', dest=key + '_unit', default=None,
            help='Unit for {0} (one of: {1}).'.format(
                analyte.label, ", ".join(u.symbol for u in analyte.reported_units)
            ),
        )
    return parser


def main():
    """Main function for command-line interface."""
    args = build_parser().parse_args()

    supplied = {
        key: (getattr(args, key), getattr(args, key + '_unit'))
        for key in CLI_ANALYTES
        if getattr(args, key) is not None
    }

    print("=" * 80)
    print("HAEMATOLOGICAL RISK PREDICTION")
    print("=" * 80)

    try:
        panel = build_panel(supplied, unit_source='operator')
    except (UnitError, ImplausibleValueError) as exc:
        print("\nRefusing to predict: {0}".format(exc))
        raise SystemExit(2)

    print("\nInput values (as given -> canonical):")
    for key, quantity in panel.measured.items():
        print("  {0:34s} {1!s:>22s} -> {2}".format(
            ANALYTES[key].label, panel.reported[key], quantity
        ))

    print("\n" + "=" * 80)
    print("Loading models...")
    print("=" * 80)

    try:
        models = load_models()
        print("Models loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train_risk_prediction_models.py first to train the models.")
        raise SystemExit(1)
    except ContractError as e:
        print(f"Refusing to serve the models: {e}")
        raise SystemExit(1)

    print("\n" + "=" * 80)
    print("Making predictions...")
    print("=" * 80)

    predictions, flags = predict_risk(models, panel)

    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    for risk_type in MODEL_TYPES:
        body = predictions[risk_type]
        outcome = body['risk'] or "not predicted ({0})".format(body.get('error', ''))
        print("\n  {0:24s} {1}".format(risk_type.capitalize() + " risk:", outcome))

    if flags:
        print("\n" + "-" * 80)
        print("Reference-range flags (CTCAE v5.0 thresholds, not a prediction)")
        print("-" * 80)
        for flag in flags:
            print("  {0}".format(flag.description))

    print("\nAssistive output. Not a diagnosis, and not a substitute for review")
    print("of the original report by a qualified clinician.")
    print("=" * 80)


if __name__ == "__main__":
    prediction_log.configure_default_handler()
    main()
