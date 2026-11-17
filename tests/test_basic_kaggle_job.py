"""Validate generated job contracts without starting a remote Kaggle run."""

import json

import pytest

from basic.kaggle_job import COMPETITION, RUNTIME_PACKAGES, SOURCE_FILES, generate_job


def test_job_is_private_cpu_and_embeds_only_runtime_sources(tmp_path):
    output = tmp_path / "job"
    result = generate_job("exampleuser", "kkbox-baseline", output, feature_modes=["baseline"])
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
    packages = {entry.split("==")[0] for entry in namespace["RUNTIME_REQUIREMENTS"]}
    assert packages == RUNTIME_PACKAGES
    assert "kaggle" not in packages and "pytest" not in packages
    assert namespace["JOB_SETTINGS"]["max_train_rows"] == 0
    assert namespace["JOB_SETTINGS"]["baseline_rounds"] == 400
    assert namespace["JOB_SETTINGS"]["enhanced_rounds"] == 600
    assert namespace["JOB_SETTINGS"]["feature_modes"] == ["baseline"]
    assert namespace["JOB_SETTINGS"]["install_dependencies"] is False
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
    result = namespace["prepare_data"](input_root, tmp_path / "prepared")
    for filename in namespace["EXPECTED_FILES"]:
        assert (result / filename).read_text() == "example\n1\n"


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
    with pytest.raises(ValueError, match="empty"):
        generate_job("exampleuser", "kkbox-baseline", tmp_path / "job")
    with pytest.raises(ValueError, match="owner"):
        generate_job("../user", "kkbox-baseline", tmp_path / "other")
