from __future__ import annotations

import sys
import threading
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest


CODE_ROOT = Path(__file__).resolve().parents[1] / "code_v2"
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

import experiment_common as common  # noqa: E402
import run_cnn_cpu_matrix as cpu_runner  # noqa: E402
import run_training_matrix as runner  # noqa: E402
import train_cnn_pn_rf as cnn  # noqa: E402


def settings(tmp_path: Path, *, model: str) -> runner.RunnerSettings:
    return runner.RunnerSettings(
        repo_root=tmp_path,
        features_root=tmp_path / "features",
        labels_root=tmp_path / "labels",
        output_root=tmp_path / "output",
        stop_file=tmp_path / "stop",
        seed=42,
        removal_runs=15,
        cnn_validation_fraction=0.2,
        cnn_test_trials=100,
        cnn_device="cuda",
        max_concurrent_experiments=(
            1 if model in {"cnn-pn-rf", "blstm"} else 4
        ),
        workers_per_experiment=(1 if model in {"cnn-pn-rf", "blstm"} else 3),
        smoke=False,
    )


def job(model: str) -> runner.ExperimentJob:
    return runner.ExperimentJob(model, "binary", 30, ("NCSU",), "UNSW1")


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_directional_splits_cover_all_four_dataset_combinations_once():
    splits = runner.directional_splits(runner.SPLIT_KINDS)

    assert len(splits) == 28
    assert len(set(splits)) == 28
    assert Counter(len(train) for train, _ in splits) == {1: 12, 2: 12, 3: 4}
    assert all(test not in train for train, test in splits)
    assert (("NCSU", "UNSW2"), "UNSW1") in splits
    assert (("NCSU", "UNSW1", "UNSW2"), "YourThing") in splits


def test_default_matrix_has_168_jobs_for_one_model():
    jobs = runner.generate_jobs(
        ["decision-tree"],
        runner.CLASSIFICATION_MODES,
        runner.WINDOW_SECONDS,
        runner.SPLIT_KINDS,
    )

    assert len(jobs) == 168
    assert len({current.job_id for current in jobs}) == 168


def test_blstm_matrix_uses_packet_lengths_and_unique_job_ids():
    jobs = runner.generate_jobs(
        ["blstm"],
        runner.CLASSIFICATION_MODES,
        runner.BLSTM_SEQUENCE_PACKETS,
        runner.SPLIT_KINDS,
    )

    assert len(jobs) == 168
    assert len({current.job_id for current in jobs}) == 168
    assert {current.window_seconds for current in jobs} == {20, 40, 100}
    assert all("sequence-" in current.job_id for current in jobs)


def test_smoke_job_uses_v2_unsw_dataset():
    jobs = runner.generate_jobs(
        ["cnn-pn-rf"],
        runner.CLASSIFICATION_MODES,
        runner.WINDOW_SECONDS,
        runner.SPLIT_KINDS,
        smoke=True,
    )

    assert jobs == [runner.ExperimentJob("cnn-pn-rf", "binary", 30, ("NCSU",), "UNSW1")]
    assert runner.generate_jobs(
        ["blstm"],
        runner.CLASSIFICATION_MODES,
        runner.BLSTM_SEQUENCE_PACKETS,
        runner.SPLIT_KINDS,
        smoke=True,
    ) == [runner.ExperimentJob("blstm", "binary", 20, ("NCSU",), "UNSW1")]


def test_removal_counts_cover_every_v2_training_dataset():
    assert runner.REMOVAL_COUNTS == {
        "NCSU": 4,
        "UNSW1": 6,
        "UNSW2": 4,
        "YourThing": 30,
    }
    three_train = runner.ExperimentJob(
        "decision-tree",
        "binary",
        30,
        ("NCSU", "UNSW1", "UNSW2"),
        "YourThing",
    )

    assert runner.removal_specs(three_train) == [
        {"dataset": "NCSU", "class": 1, "count": 4},
        {"dataset": "UNSW1", "class": 1, "count": 6},
        {"dataset": "UNSW2", "class": 1, "count": 4},
    ]
    your_thing_train = replace(three_train, train_datasets=("YourThing",))
    assert runner.removal_specs(your_thing_train) == [
        {"dataset": "YourThing", "class": 1, "count": 30}
    ]


def test_matrix_cli_requires_one_model_and_uses_v2_defaults():
    with pytest.raises(SystemExit):
        runner.parse_args([])
    with pytest.raises(SystemExit):
        runner.parse_args(["--models", "cnn-pn-rf", "decision-tree"])

    args = runner.parse_args(["--models", "decision-tree"])
    assert args.models == ["decision-tree"]
    assert args.features_root == Path("built_features_mac_v2")
    assert args.output_root == Path("output-performance-folder-mac-v2")
    assert args.windows == [5, 10, 30]
    assert args.sequence_packets == [20, 40, 100]
    assert args.classification_modes == ["binary", "multiclass"]
    assert args.cnn_rf_jobs == -1
    assert args.blstm_jobs == 1
    assert args.split_kinds == [
        "1-train-1-test",
        "2-train-1-test",
        "3-train-1-test",
    ]
    assert (
        runner.parse_args(["--models", "cnn-pn-rf", "--cnn-rf-jobs", "4"]).cnn_rf_jobs
        == 4
    )
    with pytest.raises(SystemExit):
        runner.parse_args(["--models", "cnn-pn-rf", "--cnn-rf-jobs", "0"])
    assert (
        runner.parse_args(
            [
                "--models",
                "blstm",
                "--blstm-device",
                "cpu",
                "--blstm-jobs",
                "4",
            ]
        ).blstm_jobs
        == 4
    )
    with pytest.raises(SystemExit):
        runner.parse_args(["--models", "blstm", "--blstm-jobs", "0"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--models", "blstm", "--blstm-jobs", "2"])


def test_model_specific_scheduling_policy():
    assert runner.scheduling_policy(["cnn-pn-rf"], None, 3) == (
        "cnn-pn-rf",
        runner.CNN_MAX_CONCURRENT_EXPERIMENTS,
        1,
    )
    assert runner.scheduling_policy(["decision-tree"], None, 3) == (
        "decision-tree",
        4,
        3,
    )
    assert runner.scheduling_policy(["decision-tree"], 2, 2) == (
        "decision-tree",
        2,
        2,
    )
    assert runner.scheduling_policy(["blstm"], None, 3) == ("blstm", 1, 1)
    assert runner.scheduling_policy(["blstm"], None, 3, 4, "cpu") == (
        "blstm",
        1,
        4,
    )
    assert runner.scheduling_policy(["blstm"], 2, 3, 4, "cpu") == (
        "blstm",
        2,
        4,
    )


@pytest.mark.parametrize(
    ("models", "concurrency"),
    [
        (["cnn-pn-rf", "decision-tree"], None),
        (["cnn-pn-rf"], runner.CNN_MAX_CONCURRENT_EXPERIMENTS + 1),
        (["blstm"], 2),
        (["decision-tree"], 5),
    ],
)
def test_invalid_scheduling_policy_is_rejected(models, concurrency):
    with pytest.raises(runner.MatrixError):
        runner.scheduling_policy(models, concurrency, 3)


def test_cnn_command_has_one_model_worker_and_configurable_rf_workers(tmp_path):
    command = runner.build_command(
        job("cnn-pn-rf"), settings(tmp_path, model="cnn-pn-rf")
    )

    assert command[1] == str(tmp_path / "code_v2" / "train_cnn_pn_rf.py")
    assert option_value(command, "--jobs") == "1"
    assert option_value(command, "--rf-jobs") == "-1"
    assert "taskset" not in command


def test_decision_tree_command_has_three_model_workers(tmp_path):
    command = runner.build_command(
        job("decision-tree"), settings(tmp_path, model="decision-tree")
    )

    assert command[1] == str(tmp_path / "code_v2" / "train_decision_tree.py")
    assert option_value(command, "--jobs") == "3"
    assert "taskset" not in command


def test_blstm_command_uses_packet_features_and_sequence_output(tmp_path):
    current = runner.ExperimentJob(
        "blstm", "multiclass", 40, ("NCSU",), "UNSW1"
    )
    config = replace(
        settings(tmp_path, model="blstm"),
        blstm_device="cpu",
        blstm_jobs=4,
        blstm_cache_root=tmp_path / "scratch",
    )
    command = runner.build_command(current, config)

    assert command[1] == str(tmp_path / "code_v2" / "train_lstm.py")
    assert runner.feature_root(current, config) == (
        tmp_path / "features" / "multiclass" / "packet_features"
    )
    assert runner.job_output_root(current, config).name == "sequence_40packets"
    assert option_value(command, "--model-type") == "blstm"
    assert option_value(command, "--sequence-packets") == "40"
    assert option_value(command, "--test-trials") == "100"
    assert option_value(command, "--max-sequences-per-device") == "5000"
    assert option_value(command, "--device") == "cpu"
    assert option_value(command, "--jobs") == "4"
    assert option_value(command, "--cache-root") == str(tmp_path / "scratch")


def test_cpu_cnn_runner_adds_bounded_rf_workers(tmp_path):
    config = replace(
        settings(tmp_path, model="cnn-pn-rf"),
        cnn_device="cpu",
        max_concurrent_experiments=6,
    )
    command = cpu_runner.build_cpu_command(
        job("cnn-pn-rf"),
        config,
        rf_jobs=2,
        lifecycle="resume",
    )

    assert option_value(command, "--device") == "cpu"
    assert option_value(command, "--jobs") == "1"
    assert option_value(command, "--rf-jobs") == "2"
    assert command[-1] == "--resume"


def test_cpu_cnn_runner_defaults_to_v2_windows_and_six_experiments():
    args = cpu_runner.parse_args([])

    assert args.features_root == Path("built_features_mac_v2")
    assert args.output_root == Path("output-performance-cnn-server-cpu-mac-v2")
    assert args.windows == [5, 10, 30]
    assert args.max_concurrent_experiments == 6
    assert args.cnn_rf_jobs == 2


def test_scheduling_changes_do_not_change_resume_fingerprint(tmp_path):
    matrix_jobs = [job("decision-tree")]
    current = runner.matrix_configuration(
        matrix_jobs, settings(tmp_path, model="decision-tree")
    )
    legacy = dict(current)
    legacy.pop("workers_per_experiment")
    legacy.update(
        {
            "cpu_target_percent": 85.0,
            "cpu_safety_margin_percent": 5.0,
            "cpu_ids": list(range(8)),
            "memory_target_percent": 75.0,
            "memory_per_experiment_gb": 4.0,
        }
    )

    assert common.stable_fingerprint(
        runner.fingerprint_configuration(current)
    ) == common.stable_fingerprint(runner.fingerprint_configuration(legacy))


def test_legacy_manifest_resumes_without_retaining_cpu_assignments(tmp_path):
    current_job = job("decision-tree")
    config = settings(tmp_path, model="decision-tree")
    configuration = runner.matrix_configuration([current_job], config)
    legacy_configuration = dict(configuration)
    legacy_configuration.pop("workers_per_experiment")
    legacy_configuration.update(
        {
            "cpu_target_percent": 85.0,
            "cpu_safety_margin_percent": 5.0,
            "cpu_ids": list(range(8)),
            "memory_target_percent": 75.0,
            "memory_per_experiment_gb": 4.0,
        }
    )
    fingerprint = common.stable_fingerprint(
        runner.fingerprint_configuration(legacy_configuration)
    )
    config.output_root.mkdir(parents=True)
    common.atomic_write_json(
        config.output_root / "_matrix_manifest.json",
        {
            "status": "stopped",
            "started_at": "2026-01-01T00:00:00+00:00",
            "configuration_fingerprint": fingerprint,
            "jobs": {
                current_job.job_id: {
                    "status": "pending",
                    "cpu_ids": [0, 1, 2, 3],
                }
            },
        },
    )

    _, resumed = runner.prepare_matrix_manifest(
        [current_job], config, resume=True, overwrite=False
    )

    assert "cpu_ids" not in resumed["jobs"][current_job.job_id]


def test_cnn_resume_fingerprint_ignores_only_rf_worker_allocation(tmp_path):
    current = {
        "pipeline": "cnn_pn_rf_mac_v1",
        "training": {
            "seed": 42,
            "batch_size": 1024,
            "rf_trees": 100,
            "rf_jobs": -1,
        },
    }
    legacy = {
        **current,
        "training": {**current["training"], "rf_jobs": 8},
    }
    fingerprint = common.stable_fingerprint(
        cnn.resume_fingerprint_configuration(current)
    )
    assert fingerprint == common.stable_fingerprint(
        cnn.resume_fingerprint_configuration(legacy)
    )

    experiment_dir = tmp_path / "cnn-run"
    experiment_dir.mkdir()
    common.atomic_write_json(
        experiment_dir / "_run_manifest.json",
        {"status": "stopped", "configuration_fingerprint": "legacy"},
    )
    common.atomic_write_json(
        experiment_dir / "config.json",
        {**legacy, "features_root": "features", "jobs": 1},
    )

    cnn.migrate_legacy_resource_fingerprint(
        experiment_dir, current, fingerprint, resume=True
    )

    manifest = common.read_json(experiment_dir / "_run_manifest.json")
    assert manifest["configuration_fingerprint"] == fingerprint


def test_runner_source_contains_no_resource_control_implementation():
    source = Path(runner.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "sched_getaffinity",
        "taskset",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "/proc/meminfo",
        "/proc/stat",
    ):
        assert forbidden not in source


def test_host_launchers_use_model_specific_concurrency_without_resource_limits():
    cnn_launcher = (REPO_ROOT / "run-cnn-wsl.sh").read_text(encoding="utf-8")
    tree_launcher = (REPO_ROOT / "run-decision-tree-server.sh").read_text(
        encoding="utf-8"
    )

    assert "--max-concurrent-experiments 1" in cnn_launcher
    assert "code_v2/run_training_matrix.py" in cnn_launcher
    assert "output-performance-cnn-mac-v2" in cnn_launcher
    assert "--rf-jobs" not in cnn_launcher
    assert "memory-per-experiment" not in cnn_launcher
    assert "--max-concurrent-experiments 4" in tree_launcher
    assert "code_v2/run_training_matrix.py" in tree_launcher
    assert "output-performance-decision-tree-mac-v2" in tree_launcher
    assert "--decision-tree-jobs 3" in tree_launcher
    assert "memory-per-experiment" not in tree_launcher


def test_v1_launchers_are_archived_and_v2_monitor_is_used():
    archive = REPO_ROOT / "v1-feature-training-and-validation"
    for name in (
        "run-cnn-wsl.sh",
        "run-decision-tree-server.sh",
        "smoke-run-training-matrix.sh",
        "monitor-training-job.sh",
    ):
        assert (archive / name).is_file()

    monitor_launcher = (REPO_ROOT / "monitor-training-job.sh").read_text(
        encoding="utf-8"
    )
    smoke_launcher = (REPO_ROOT / "smoke-run-training-matrix.sh").read_text(
        encoding="utf-8"
    )
    assert "code_v2/monitor_training_matrix.py" in monitor_launcher
    assert "-v2" in monitor_launcher
    assert "-v2" in smoke_launcher


def test_decision_tree_scheduler_never_exceeds_four_experiments(tmp_path, monkeypatch):
    matrix_jobs = [
        runner.ExperimentJob("decision-tree", "binary", window, ("NCSU",), "UNSW1")
        for window in range(1, 7)
    ]
    config = settings(tmp_path, model="decision-tree")
    active = 0
    maximum_active = 0
    lock = threading.Lock()

    monkeypatch.setattr(runner, "validate_job_inputs", lambda *args: None)
    monkeypatch.setattr(runner, "trainer_status", lambda *args: None)

    def fake_run_job(current_job, current_settings, *, overwrite):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return (
            "complete",
            0,
            [],
            runner.expected_experiment_directory(current_job, current_settings),
        )

    monkeypatch.setattr(runner, "run_job", fake_run_job)

    result = runner.execute_matrix(
        matrix_jobs,
        config,
        resume=False,
        overwrite=False,
        dry_run=False,
        continue_on_error=False,
    )

    assert result == 0
    assert maximum_active == 4
    manifest = common.read_json(config.output_root / "_matrix_manifest.json")
    assert manifest["runtime_allocation"] == {
        "concurrent_experiments": 4,
        "workers_per_experiment": 3,
        "cpu_affinity": "unrestricted",
        "memory_limit": "unrestricted",
    }
