"""Validate generated job contracts without starting a remote Kaggle run."""

import json
import re

import pytest

from basic.kaggle_job import COMPETITION, RUNTIME_PACKAGES, SOURCE_FILES, generate_job


@pytest.mark.parametrize("disable_frequency_features", [False, True])
def test_job_is_private_cpu_and_embeds_only_runtime_sources(tmp_path, disable_frequency_features):
    output = tmp_path / "job"
    result = generate_job("exampleuser", "kkbox-baseline", output, feature_modes=["baseline"],
                          disable_frequency_features=disable_frequency_features,
                          user_context_features=disable_frequency_features)
    metadata = json.loads((output / "kernel-metadata.json").read_text())
    assert metadata["id"] == "exampleuser/kkbox-baseline"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["kernel_type"] == "script"
    assert metadata["competition_sources"] == [COMPETITION]
    script = (output / metadata["code_file"]).read_text(encoding="utf-8")
    namespace = {"__name__": "generated_script_for_test"}
    # main is guarded; evaluating definitions cannot install or start training.
    exec(compile(script, "run.py", "exec"), namespace)
    assert set(namespace["SOURCES"]) == set(SOURCE_FILES)
    packages = {re.split(r"[<>=!~\[]", entry, maxsplit=1)[0] for entry in namespace["RUNTIME_REQUIREMENTS"]}
    assert packages == RUNTIME_PACKAGES
    assert "kaggle" not in packages and "pytest" not in packages
    assert namespace["JOB_SETTINGS"]["max_train_rows"] == 0
    assert namespace["JOB_SETTINGS"]["baseline_rounds"] == 400
    assert namespace["JOB_SETTINGS"]["enhanced_rounds"] == 600
    assert namespace["JOB_SETTINGS"]["feature_modes"] == ["baseline"]
    assert namespace["JOB_SETTINGS"]["install_dependencies"] is False
    assert namespace["JOB_SETTINGS"]["disable_frequency_features"] is disable_frequency_features
    assert namespace["JOB_SETTINGS"]["user_context_features"] is disable_frequency_features
    assert len(result["run_script_sha256"]) == 64


def test_generated_data_preparation_handles_official_archives(tmp_path, monkeypatch):
    import py7zr
    output = tmp_path / "job"
    generate_job("exampleuser", "kkbox-baseline", output)
    namespace = {"__name__": "generated_script_for_test"}
    exec((output / "run.py").read_text(encoding="utf-8"), namespace)
    monkeypatch.setattr(namespace["shutil"], "which", lambda command: None)
    input_root = tmp_path / "input"
    input_root.mkdir()
    source = tmp_path / "source.csv"
    source.write_text("example\n1\n", encoding="utf-8")
    for filename in namespace["EXPECTED_FILES"]:
        with py7zr.SevenZipFile(input_root / (filename + ".7z"), mode="w") as archive:
            archive.write(source, arcname=filename)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    # Earlier versions wrote directly to the final path before extraction ended.
    (prepared / "train.csv").write_text("old truncated nonempty CSV")
    result = namespace["prepare_data"](input_root, prepared)
    for filename in namespace["EXPECTED_FILES"]:
        assert (result / filename).read_text() == "example\n1\n"
    namespace["prepare_data"](input_root, result)
    assert not list(result.glob("*.partial"))


def test_system_seven_zip_is_used_without_py7zr(tmp_path, monkeypatch):
    from types import SimpleNamespace
    output = tmp_path / "job"
    generate_job("exampleuser", "kkbox-baseline", output)
    namespace = {"__name__": "generated_script_for_test"}
    exec((output / "run.py").read_text(encoding="utf-8"), namespace)
    monkeypatch.setattr(namespace["shutil"], "which", lambda command: "/usr/bin/7z" if command == "7z" else None)
    calls = []
    def fake_run(arguments, **kwargs):
        calls.append(arguments)
        if arguments[1] == "l":
            return SimpleNamespace(stdout="Path = archive.7z\n\n----------\nPath = train.csv\nSize = 2\n")
        kwargs["stdout"].write(b"a\n")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(namespace["subprocess"], "run", fake_run)
    destination = tmp_path / "train.csv"
    namespace["extract_archive"](tmp_path / "train.csv.7z", "train.csv", destination)
    assert destination.read_bytes() == b"a\n"
    assert calls[0][1:3] == ["l", "-slt"]
    assert calls[1][1:5] == ["e", "-so", "-bd", "-y"]


def test_runtime_check_and_explicit_install_option(tmp_path):
    output = tmp_path / "job"
    generate_job("exampleuser", "kkbox-baseline", output, install_dependencies=True)
    metadata = json.loads((output / "kernel-metadata.json").read_text())
    assert metadata["enable_internet"] is True
    namespace = {"__name__": "generated_script_for_test"}
    exec((output / "run.py").read_text(encoding="utf-8"), namespace)
    versions = namespace["check_runtime_dependencies"]()
    assert set(versions) == RUNTIME_PACKAGES - {"py7zr"}
    assert namespace["JOB_SETTINGS"]["install_dependencies"] is True


def test_generator_rejects_overwrite_and_invalid_identity(tmp_path):
    generate_job("exampleuser", "kkbox-baseline", tmp_path / "job")
    with pytest.raises(ValueError, match="overwrite"):
        generate_job("exampleuser", "kkbox-baseline", tmp_path / "job")
    (tmp_path / "job" / "notes.txt").write_text("keep")
    generate_job("exampleuser", "kkbox-baseline", tmp_path / "job", overwrite=True,
        numeric_features="legacy", context_counting="full", user_preference_features=True,
        collaborative_features=True, split_mode="row_order", no_refit=True, num_leaves=127, max_cat_threshold=64)
    assert (tmp_path / "job" / "notes.txt").read_text() == "keep"
    settings = json.loads((tmp_path / "job" / "job_configuration.json").read_text())["settings"]
    assert settings["numeric_features"] == "legacy"
    assert settings["context_counting"] == "full"
    assert settings["user_preference_features"] is True
    assert settings["collaborative_features"] is True
    assert settings["split_mode"] == "row_order"
    assert settings["no_refit"] is True
    assert settings["num_leaves"] == 127 and settings["max_cat_threshold"] == 64
    with pytest.raises(ValueError, match="owner"):
        generate_job("../user", "kkbox-baseline", tmp_path / "other")


def test_generated_runner_passes_transductive_encoding_and_capacity_options(tmp_path):
    output = tmp_path / "job"
    generate_job("exampleuser", "kkbox-transductive", output, feature_modes=["enhanced"],
                 transductive=True, row_order_features=True, target_encoding=True,
                 collaborative_features=True, song_components=48, artist_components=24, genre_components=16,
                 no_refit=True, split_mode="row_order", num_leaves=255, learning_rate=0.05,
                 min_child_samples=100, max_cat_threshold=64, feature_fraction=0.8)
    namespace = {"__name__": "generated_script_for_test"}
    exec((output / "run.py").read_text(encoding="utf-8"), namespace)
    working = tmp_path / "working"
    namespace["WORKING_ROOT"] = working
    namespace["prepare_data"] = lambda: tmp_path / "data"
    namespace["check_runtime_dependencies"] = lambda: {}
    calls = []

    def run(arguments, *unused):
        calls.append(arguments)
        if arguments[2] == "basic.train":
            directory = working / "enhanced"
            directory.mkdir()
            (directory / "metrics.json").write_text("{}")

    namespace["run_checked"] = run
    namespace["main"]()
    training = calls[0]
    for flag in ("--transductive", "--row-order-features", "--target-encoding", "--collaborative-features", "--no-refit"):
        assert flag in training
    for flag, value in {"--song-components": "48", "--artist-components": "24", "--genre-components": "16",
                        "--num-leaves": "255", "--min-child-samples": "100", "--max-cat-threshold": "64",
                        "--feature-fraction": "0.8", "--learning-rate": "0.05", "--split-mode": "row_order"}.items():
        assert training[training.index(flag) + 1] == value
    assert (working / "code" / "basic" / "target_encoding.py").is_file()
