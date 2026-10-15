# Copyright 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The published test arm is the builder's output, a JBeam vehicle mod."""

import hashlib
import json
import zipfile
from pathlib import Path

from openbeam_gym.env import TEST_ARM_FILE, TEST_ARM_SHA256, SceneConfig
from openbeam_gym.robots import build_test_arm

ASSET = Path(__file__).resolve().parents[1] / "assets" / "openbeam_test_arm.zip"


def test_the_published_zip_is_the_builders_output(tmp_path):
    out = tmp_path / "arm.zip"
    digest = build_test_arm.write_zip(out, build_test_arm.members())
    assert digest == TEST_ARM_SHA256
    assert hashlib.sha256(ASSET.read_bytes()).hexdigest() == TEST_ARM_SHA256


def test_it_is_a_jbeam_vehicle_package_with_a_robot_spec():
    with zipfile.ZipFile(ASSET) as z:
        names = set(z.namelist())
        assert not any(n.endswith((".truck", ".load", ".fixed")) for n in names)
        assert "vehicles/openbeam_test_arm/openbeam_test_arm.jbeam" in names
        jbeam = json.loads(z.read("vehicles/openbeam_test_arm/openbeam_test_arm.jbeam"))
        spec = json.loads(z.read("vehicles/openbeam_test_arm/openbeam_robot.json"))
    part = jbeam["openbeam_test_arm"]
    node_ids = {row[0] for row in part["nodes"][1:] if isinstance(row, list)}
    assert spec["schema"] == "openbeam-robot/1"
    assert [j["name"] for j in spec["joints"]] == ["base_pan", "shoulder", "elbow", "wrist_roll"]
    sources = {row[4]["inputSource"] for row in part["torsionHydros"][1:]}
    assert sources == {f"arm_cmd_{j['name']}" for j in spec["joints"]}
    for j in spec["joints"]:
        assert set(j["axis"]) | {j["parent_ref"], j["child_ref"]} <= node_ids
        assert j["lower"] < 0 < j["upper"]
    assert "tcp" in node_ids
    assert SceneConfig().robot_file == TEST_ARM_FILE
