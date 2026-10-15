# assets

`openbeam_test_arm.zip`: the fast test robot, a four-joint arm (`base_pan`,
`shoulder`, `elbow`, `wrist_roll`, with a `tcp` tool node) packaged as a JBeam
vehicle mod (`vehicles/openbeam_test_arm/`). The game's `install_mod` takes it
as it is; no classic `.truck` format is involved. `SceneConfig()` installs it
with `install_mod {"url": <raw URL of this file>, "sha256": <its SHA-256>}`
(`openbeam_gym.env.TEST_ARM_URL` / `TEST_ARM_SHA256`) and spawns
`openbeam_test_arm.jbeam`.

- Links are stiff node clusters; each joint is a hinge on two axis nodes,
  driven by four JBeam `torsionHydros` whose rest angle follows the electric
  `arm_cmd_<joint>` (radians from the spawn pose, 1 rad/s).
- `openbeam_robot.json` (`openbeam-robot/1`) declares the joints, limits and
  target electrics for the game's robot tools; the vehicle Lua publishes
  `robot_dof`, `robot_q<i>`, `robot_qd<i>`, `robot_cmd<i>` and
  `arm_<joint>_pos|vel|target`, and binds keys (A/D, W/S, Q/E, Z/X, 3 home)
  that also make it a seat for the player, which stepped physics needs.
- It is a soft body: expect a few centimetres of sag under gravity, which
  `robot_fk` reports as `simulated` error.

Rebuild it (deterministic, stdlib only):

```sh
python -m openbeam_gym.robots.build_test_arm --out assets/openbeam_test_arm.zip
```

and update `TEST_ARM_SHA256` in `openbeam_gym/env.py` when it changes.
