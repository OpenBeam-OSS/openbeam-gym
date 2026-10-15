"""Write the openbeam-robot/1 spec of OpenBeam's SO-101 from its node-model generator.

    python -m openbeam_gym.robots.make_spec_from_so101_model <dir with so101_model.py> > so101.json

Reads the generator's so101_model.build(): hinge node
pairs, reference nodes, axis signs, URDF limits, the base frame nodes and the
gripper_frame_link pose in its node basis, so robot_fk/robot_ik answer in the
URDF's base_link frame. The command and state channels are SO101.md's
electrics. Ship the result as vehicles/so101/openbeam_robot.json, or pass it
to robot_attach.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def spec_from_model(model) -> dict:
    import numpy as np

    def frame(o, x, y):
        p = [model.nodes[n].pos for n in (o, x, y)]
        ex = (p[1] - p[0]) / np.linalg.norm(p[1] - p[0])
        v = p[2] - p[0]
        ey = v - ex * (ex @ v)
        ey /= np.linalg.norm(ey)
        return np.column_stack([ex, ey, np.cross(ex, ey)]), p[0]

    # Base frame: base_link is the jbeam's authoring frame. Express it in the
    # basis of three base nodes so it follows the pinned base.
    bo, bx, by = model.base_frame_nodes[0], "bfront", "bleft"
    basis, origin = frame(bo, bx, by)
    base_rotation = basis.T            # base_link axes in the node basis
    base_offset = basis.T @ (np.zeros(3) - origin)
    joints = []
    for js in model.joints:
        joints.append({"name": js.name, "axis": list(js.hinge), "parent_ref": js.parent_ref,
                       "child_ref": js.child_ref, "sign": js.sign, "lower": js.lower, "upper": js.upper,
                       "reference": 0.0})
    return {
        "schema": "openbeam-robot/1",
        "name": "so101",
        "source": "OpenBeam SO-101 node model, URDF so101_new_calib",
        "base_frame": {"nodes": [bo, bx, by], "offset": base_offset.tolist(),
                       "rotation": base_rotation.reshape(-1).tolist()},
        "joints": joints,
        "end_effectors": [{"name": "gripper_frame_link", "nodes": list(model.ee_frame_nodes),
                           "offset": np.asarray(model.ee_offset).tolist(),
                           "rotation": np.asarray(model.ee_rotation).reshape(-1).tolist()}],
        "targets": {"lua": "so101.setJoints"},
        "state": {"position": "so101_{joint}_pos", "velocity": "so101_{joint}_vel",
                  "effort": "so101_{joint}_effort", "target": "so101_{joint}_target",
                  "limit": "so101_{joint}_limit"},
    }


def main() -> None:
    tools = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(tools))
    import so101_model  # noqa: E402

    print(json.dumps(spec_from_model(so101_model.build()), indent=1))


if __name__ == "__main__":
    main()
