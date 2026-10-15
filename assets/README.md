# assets

`openbeam_test_arm.zip`: a four-joint rotator arm in OpenBeam's classic truck
format (`base_pan`, `shoulder`, `elbow`, `wrist_roll`, with a `tcp` tool node),
used by the examples and the live checks. Install it in a game with
`install_mod {"url": <raw URL of this file>, "sha256": <its SHA-256>}`, then
`spawn_vehicle {"file": "openbeam_test_arm.truck"}`. It is a compliant soft body:
expect centimetre-scale sag, which `robot_fk` reports as `simulated` error.
