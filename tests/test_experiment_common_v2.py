from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_v2 import experiment_common as common


DATASET = "NCSU"
DEVICE = "02:00:00:00:00:01"


def write_contract_fixture(tmp_path, *, algorithm_version=2, configuration=None):
    features = tmp_path / "features"
    labels = tmp_path / "labels"
    features.mkdir(parents=True)
    label_dir = labels / DATASET
    label_dir.mkdir(parents=True)
    label_bytes = json.dumps({DEVICE: {"is_iot_device": 1}}).encode()
    (label_dir / common.DEFAULT_MAC_LABEL_FILE).write_bytes(label_bytes)
    contract = {
        "label_file": common.DEFAULT_MAC_LABEL_FILE,
        "model_feature_columns": list(common.FEATURE_COLUMNS),
        "nullable_feature_columns": list(common.NULLABLE_FEATURE_COLUMNS),
        "missing_indicator_columns": list(common.MISSING_INDICATOR_COLUMNS),
    }
    if configuration:
        contract.update(configuration)
    (features / "_run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "algorithm_version": algorithm_version,
                "configuration": contract,
                "datasets": {
                    DATASET: {
                        "identifier_type": "mac",
                        "label_sha256": hashlib.sha256(label_bytes).hexdigest(),
                    }
                },
            }
        )
    )
    return features, labels


def feature_frame() -> pd.DataFrame:
    return pd.DataFrame([{column: 1.0 for column in common.FEATURE_COLUMNS}])


def test_v2_manifest_contract_is_exact(tmp_path):
    features, labels = write_contract_fixture(tmp_path)
    provenance = common.validate_mac_feature_manifest(
        features,
        labels,
        common.DEFAULT_MAC_LABEL_FILE,
        [DATASET],
    )
    assert provenance["algorithm_version"] == 2
    assert provenance["model_feature_columns"] == list(common.MODEL_FEATURE_COLUMNS)


def test_v2_manifest_rejects_v1_and_changed_column_order(tmp_path):
    features, labels = write_contract_fixture(tmp_path, algorithm_version=1)
    with pytest.raises(common.PipelineError, match="unsupported feature algorithm"):
        common.validate_mac_feature_manifest(
            features, labels, common.DEFAULT_MAC_LABEL_FILE, [DATASET]
        )

    features, labels = write_contract_fixture(
        tmp_path / "changed",
        configuration={
            "model_feature_columns": list(reversed(common.FEATURE_COLUMNS))
        },
    )
    with pytest.raises(common.PipelineError, match="model_feature_columns"):
        common.validate_mac_feature_manifest(
            features, labels, common.DEFAULT_MAC_LABEL_FILE, [DATASET]
        )


def test_nullable_features_become_zero_values_and_fixed_indicators():
    frame = feature_frame()
    frame.loc[0, list(common.NULLABLE_FEATURE_COLUMNS)] = np.nan
    matrix = common.raw_feature_matrix(frame)
    assert matrix.shape == (1, 22)
    nullable_indexes = [
        common.FEATURE_COLUMNS.index(column)
        for column in common.NULLABLE_FEATURE_COLUMNS
    ]
    np.testing.assert_array_equal(matrix[0, nullable_indexes], 0.0)
    np.testing.assert_array_equal(matrix[0, len(common.FEATURE_COLUMNS) :], 1.0)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("outgoing_packet_count", np.nan, "required feature"),
        ("inter_arrival_cv", np.inf, "infinite"),
        ("active_span", -1.0, "nonnegative"),
    ],
)
def test_invalid_model_values_are_rejected(column, value, message):
    frame = feature_frame()
    frame.loc[0, column] = value
    with pytest.raises(common.PipelineError, match=message):
        common.raw_feature_matrix(frame)


def test_log_transform_preserves_missing_indicators():
    frame = feature_frame()
    frame.loc[0, common.NULLABLE_FEATURE_COLUMNS[0]] = np.nan
    matrix = common.raw_feature_matrix(frame)
    transformed = common.log_model_features(matrix)
    np.testing.assert_allclose(
        transformed[0, : len(common.FEATURE_COLUMNS)],
        np.log1p(matrix[0, : len(common.FEATURE_COLUMNS)]),
    )
    np.testing.assert_array_equal(
        transformed[0, len(common.FEATURE_COLUMNS) :],
        matrix[0, len(common.FEATURE_COLUMNS) :],
    )
