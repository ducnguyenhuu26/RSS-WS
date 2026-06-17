from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DimensionSpec:
    index: int
    name: str
    kind: str
    role: str
    law_hint: str
    constraints: str = ""
    related: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionSpec:
    index: int
    name: str
    kind: str
    affects: tuple[str, ...]
    law_hint: str = ""
    bounds: str = "usually normalized to [-1, 1]"


@dataclass(frozen=True)
class MujocoTaskSpec:
    env_id: str
    state_dimensions: tuple[DimensionSpec, ...]
    action_dimensions: tuple[ActionSpec, ...]
    known_structure: tuple[str, ...]
    safe_law_families: tuple[str, ...]
    forbidden_law_families: tuple[str, ...]
    reward_semantics: str
    termination_semantics: str


def get_mujoco_task_spec(
    env_id: str,
    state_dim: int | None = None,
    action_dim: int | None = None,
) -> MujocoTaskSpec:
    spec = _TASK_SPECS.get(_normalize_env_id(env_id))
    if spec is None:
        return _fallback_task_spec(env_id, state_dim=state_dim, action_dim=action_dim)
    if state_dim is not None and state_dim != len(spec.state_dimensions):
        return _fallback_task_spec(env_id, state_dim=state_dim, action_dim=action_dim)
    if action_dim is not None and action_dim != len(spec.action_dimensions):
        return _fallback_task_spec(env_id, state_dim=state_dim, action_dim=action_dim)
    return spec


def format_task_spec_for_prompt(spec: MujocoTaskSpec) -> str:
    state_lines = []
    for dim in spec.state_dimensions:
        related = f"; related={', '.join(dim.related)}" if dim.related else ""
        constraints = f"; constraints={dim.constraints}" if dim.constraints else ""
        state_lines.append(
            f"- state[{dim.index}] {dim.name}: kind={dim.kind}; role={dim.role}; "
            f"law_hint={dim.law_hint}{constraints}{related}"
        )

    action_lines = []
    for action in spec.action_dimensions:
        affects = ", ".join(action.affects) if action.affects else "task-dependent"
        action_lines.append(
            f"- action[{action.index}] {action.name}: kind={action.kind}; "
            f"bounds={action.bounds}; affects={affects}; law_hint={action.law_hint}"
        )
    mujoco_group_lines = _format_mujoco_coordinate_groups(spec)

    return "\n".join(
        [
            "TASK-CONDITIONED MUJOCO SEMANTICS",
            f"Environment: {spec.env_id}",
            "",
            "MUJOCO COORDINATE GROUPS",
            *mujoco_group_lines,
            "",
            "OBSERVATION / STATE DIMENSION SEMANTICS",
            *state_lines,
            "",
            "ACTION DIMENSION SEMANTICS",
            *action_lines,
            "",
            "KNOWN STRUCTURE",
            *[f"- {item}" for item in spec.known_structure],
            "",
            "SAFE LAW FAMILIES",
            *[f"- {item}" for item in spec.safe_law_families],
            "",
            "FORBIDDEN / HIGH-RISK LAW FAMILIES",
            *[f"- {item}" for item in spec.forbidden_law_families],
            "",
            "REWARD AND PLANNING SEMANTICS",
            f"- reward: {spec.reward_semantics}",
            f"- termination: {spec.termination_semantics}",
        ]
    )


def _format_mujoco_coordinate_groups(spec: MujocoTaskSpec) -> list[str]:
    grouped_states: dict[str, list[str]] = {
        "qpos / generalized position-like coordinates": [],
        "qvel / generalized velocity coordinates": [],
        "qfrc_constraint / constraint-force coordinates": [],
        "derived or exogenous observation coordinates": [],
    }
    for dim in spec.state_dimensions:
        kind = dim.kind.lower()
        item = f"state[{dim.index}]={dim.name}"
        if "qvel" in kind:
            grouped_states["qvel / generalized velocity coordinates"].append(item)
        elif "constraint" in kind or "qfrc" in kind:
            grouped_states["qfrc_constraint / constraint-force coordinates"].append(item)
        elif "qpos" in kind or kind in {"sin_angle", "cos_angle"}:
            grouped_states["qpos / generalized position-like coordinates"].append(item)
        else:
            grouped_states["derived or exogenous observation coordinates"].append(item)

    lines: list[str] = []
    for label, values in grouped_states.items():
        if values:
            lines.append(f"- {label}: {', '.join(values)}")

    torque_like = [
        f"action[{action.index}]={action.name}"
        for action in spec.action_dimensions
        if action.kind.lower() in {"torque", "force"}
    ]
    other_actions = [
        f"action[{action.index}]={action.name}"
        for action in spec.action_dimensions
        if action.kind.lower() not in {"torque", "force"}
    ]
    if torque_like:
        lines.append(
            "- ctrl / actuator force-torque inputs: "
            f"{', '.join(torque_like)}. These are action controls, not qpos/qvel; "
            "in MuJoCo they induce generalized forces through the actuator model."
        )
    if other_actions:
        lines.append(f"- other action controls: {', '.join(other_actions)}")
    if not lines:
        lines.append("- unknown MuJoCo coordinate grouping; use conservative laws only")
    return lines


def _normalize_env_id(env_id: str) -> str:
    return str(env_id).strip()


def _fallback_task_spec(
    env_id: str,
    state_dim: int | None,
    action_dim: int | None,
) -> MujocoTaskSpec:
    state_count = int(state_dim or 0)
    action_count = int(action_dim or 0)
    return MujocoTaskSpec(
        env_id=env_id,
        state_dimensions=tuple(
            DimensionSpec(
                index=index,
                name=f"unknown_state_{index}",
                kind="unknown",
                role="no verified MuJoCo semantic label is registered",
                law_hint="leave to neural residual unless a transition pattern is very strong",
            )
            for index in range(state_count)
        ),
        action_dimensions=tuple(
            ActionSpec(
                index=index,
                name=f"unknown_action_{index}",
                kind="unknown_control",
                affects=(),
                law_hint="use only if samples show a robust sparse effect",
            )
            for index in range(action_count)
        ),
        known_structure=(
            "No registered task spec is available. Treat coordinate semantics as unknown.",
        ),
        safe_law_families=(
            "very sparse identity or local linear corrections with low confidence",
        ),
        forbidden_law_families=(
            "broad laws that overwrite every dimension",
            "laws that assume positions and velocities are split exactly in half",
        ),
        reward_semantics="unknown",
        termination_semantics="unknown",
    )


_TASK_SPECS: dict[str, MujocoTaskSpec] = {
    "InvertedPendulum-v5": MujocoTaskSpec(
        env_id="InvertedPendulum-v5",
        state_dimensions=(
            DimensionSpec(0, "cart_position", "qpos", "cart horizontal position", "q_next = q + dt * qdot", related=("state[2]",)),
            DimensionSpec(1, "pole_angle", "qpos_angle", "pole angle from upright", "angle_next = angle + dt * angular_velocity", constraints="termination-sensitive: keep angle semantics exact", related=("state[3]",)),
            DimensionSpec(2, "cart_velocity", "qvel", "cart horizontal velocity", "action may affect acceleration", related=("state[0]", "action[0]")),
            DimensionSpec(3, "pole_angular_velocity", "qvel", "pole angular velocity", "action and angle may affect angular acceleration", related=("state[1]", "action[0]")),
        ),
        action_dimensions=(
            ActionSpec(0, "cart_force", "force", ("state[2]", "state[3]"), "prefer sparse action-to-velocity effects"),
        ),
        known_structure=(
            "Observation is two generalized coordinates followed by two generalized velocities.",
            "The safest symbolic laws are kinematic updates from qvel to qpos.",
        ),
        safe_law_families=(
            "cart_position_next = cart_position + dt * cart_velocity",
            "pole_angle_next = pole_angle + dt * pole_angular_velocity",
            "low-confidence local linear acceleration laws using action[0]",
        ),
        forbidden_law_families=(
            "do not overwrite both velocities with constant offsets",
            "do not predict termination or reward directly inside F_program",
        ),
        reward_semantics="keep the pole upright and cart near the valid region",
        termination_semantics="terminates when the pole angle leaves the healthy range",
    ),
    "InvertedDoublePendulum-v5": MujocoTaskSpec(
        env_id="InvertedDoublePendulum-v5",
        state_dimensions=(
            DimensionSpec(0, "cart_position", "qpos", "cart horizontal position", "q_next = q + dt * qdot", related=("state[5]",)),
            DimensionSpec(1, "sin_pole1_angle", "sin_angle", "sine encoding of first pole angle", "avoid direct linear q_next update; use paired cos/sin consistency", constraints="paired with state[3]", related=("state[3]", "state[6]")),
            DimensionSpec(2, "sin_pole2_angle", "sin_angle", "sine encoding of second pole angle", "avoid direct linear q_next update; use paired cos/sin consistency", constraints="paired with state[4]", related=("state[4]", "state[7]")),
            DimensionSpec(3, "cos_pole1_angle", "cos_angle", "cosine encoding of first pole angle", "avoid independent linear drift", constraints="paired with state[1]", related=("state[1]", "state[6]")),
            DimensionSpec(4, "cos_pole2_angle", "cos_angle", "cosine encoding of second pole angle", "avoid independent linear drift", constraints="paired with state[2]", related=("state[2]", "state[7]")),
            DimensionSpec(5, "cart_velocity", "qvel", "cart horizontal velocity", "action may affect acceleration", related=("state[0]", "action[0]")),
            DimensionSpec(6, "pole1_angular_velocity", "qvel", "first pole angular velocity", "angle and action may affect acceleration", related=("state[1]", "state[3]", "action[0]")),
            DimensionSpec(7, "pole2_angular_velocity", "qvel", "second pole angular velocity", "angle and action may affect acceleration", related=("state[2]", "state[4]", "action[0]")),
            DimensionSpec(8, "constraint_force_x", "constraint", "included constraint/contact-like force summary", "usually leave to neural residual", constraints="do not use as a stable state coordinate unless evidence is strong"),
        ),
        action_dimensions=(
            ActionSpec(0, "cart_force", "force", ("state[5]", "state[6]", "state[7]"), "sparse action-to-velocity effects only"),
        ),
        known_structure=(
            "Angles are encoded as sin/cos pairs, not raw angles.",
            "Only state[0] is a raw position with a direct velocity coordinate state[5].",
            "The last coordinate is constraint-like and risky for hand-written dynamics.",
        ),
        safe_law_families=(
            "cart_position_next = cart_position + dt * cart_velocity",
            "first-order sin/cos updates only if both paired coordinates are updated coherently",
            "low-confidence sparse velocity laws using action[0]",
        ),
        forbidden_law_families=(
            "do not update sin and cos coordinates independently with arbitrary linear drift",
            "do not force a half-position/half-velocity split",
            "do not predict constraint coordinates with broad deterministic formulas",
        ),
        reward_semantics="stabilize the double pendulum upright while keeping the tip high",
        termination_semantics="terminates when the pendulum tip falls below the healthy threshold",
    ),
    "Reacher-v5": MujocoTaskSpec(
        env_id="Reacher-v5",
        state_dimensions=(
            DimensionSpec(0, "cos_joint0_angle", "cos_angle", "cosine of shoulder joint angle", "do not update independently from state[2]", constraints="paired with state[2]", related=("state[2]", "state[6]")),
            DimensionSpec(1, "cos_joint1_angle", "cos_angle", "cosine of elbow joint angle", "do not update independently from state[3]", constraints="paired with state[3]", related=("state[3]", "state[7]")),
            DimensionSpec(2, "sin_joint0_angle", "sin_angle", "sine of shoulder joint angle", "do not update independently from state[0]", constraints="paired with state[0]", related=("state[0]", "state[6]")),
            DimensionSpec(3, "sin_joint1_angle", "sin_angle", "sine of elbow joint angle", "do not update independently from state[1]", constraints="paired with state[1]", related=("state[1]", "state[7]")),
            DimensionSpec(4, "target_x", "target", "target x coordinate", "constant within an episode; usually identity", constraints="do not treat as dynamics"),
            DimensionSpec(5, "target_y", "target", "target y coordinate", "constant within an episode; usually identity", constraints="do not treat as dynamics"),
            DimensionSpec(6, "joint0_velocity", "qvel", "shoulder angular velocity", "action[0] may affect acceleration", related=("action[0]",)),
            DimensionSpec(7, "joint1_velocity", "qvel", "elbow angular velocity", "action[1] may affect acceleration", related=("action[1]",)),
            DimensionSpec(8, "fingertip_to_target_x", "geometry", "x distance from fingertip to target", "derived geometry; leave to neural residual unless very confident", related=("state[0]", "state[1]", "state[2]", "state[3]", "state[4]")),
            DimensionSpec(9, "fingertip_to_target_y", "geometry", "y distance from fingertip to target", "derived geometry; leave to neural residual unless very confident", related=("state[0]", "state[1]", "state[2]", "state[3]", "state[5]")),
        ),
        action_dimensions=(
            ActionSpec(0, "shoulder_torque", "torque", ("state[6]",), "sparse local action effect"),
            ActionSpec(1, "elbow_torque", "torque", ("state[7]",), "sparse local action effect"),
        ),
        known_structure=(
            "The first four coordinates are two sin/cos angle pairs.",
            "Target coordinates are exogenous episode constants.",
            "Distance-to-target coordinates are derived reward-relevant geometry, not independent physics states.",
        ),
        safe_law_families=(
            "identity laws for target coordinates",
            "paired sin/cos updates if both coordinates are kept consistent",
            "sparse torque-to-angular-velocity laws",
        ),
        forbidden_law_families=(
            "do not directly push target coordinates with action",
            "do not independently update one side of a sin/cos pair",
            "do not claim derived fingertip distances are independent qpos variables",
        ),
        reward_semantics="negative distance to target plus action-control penalty",
        termination_semantics="does not naturally terminate before truncation",
    ),
    "Swimmer-v5": MujocoTaskSpec(
        env_id="Swimmer-v5",
        state_dimensions=(
            DimensionSpec(0, "front_tip_angle", "qpos_angle", "angle of the front tip", "angle_next = angle + dt * angular_velocity", related=("state[5]",)),
            DimensionSpec(1, "motor1_angle", "qpos_angle", "angle of the first rotor", "angle_next = angle + dt * angular_velocity", related=("state[6]", "action[0]")),
            DimensionSpec(2, "motor2_angle", "qpos_angle", "angle of the second rotor", "angle_next = angle + dt * angular_velocity", related=("state[7]", "action[1]")),
            DimensionSpec(3, "front_tip_x_velocity", "qvel", "velocity of the tip along x; reward-relevant", "action-conditioned velocity dynamics", related=("action[0]", "action[1]")),
            DimensionSpec(4, "front_tip_y_velocity", "qvel", "velocity of the tip along y", "action-conditioned velocity dynamics", related=("action[0]", "action[1]")),
            DimensionSpec(5, "front_tip_angular_velocity", "qvel", "angular velocity of the front tip", "local angular acceleration", related=("state[0]",)),
            DimensionSpec(6, "motor1_angular_velocity", "qvel", "angular velocity of the first rotor", "action[0] may affect acceleration", related=("state[1]", "action[0]")),
            DimensionSpec(7, "motor2_angular_velocity", "qvel", "angular velocity of the second rotor", "action[1] may affect acceleration", related=("state[2]", "action[1]")),
        ),
        action_dimensions=(
            ActionSpec(0, "motor1_torque", "torque", ("state[6]", "state[3]", "state[4]"), "sparse torque-to-velocity effects"),
            ActionSpec(1, "motor2_torque", "torque", ("state[7]", "state[3]", "state[4]"), "sparse torque-to-velocity effects"),
        ),
        known_structure=(
            "Global x/y positions are excluded from the observation.",
            "The first three coordinates are position-like angles, followed by five velocities.",
        ),
        safe_law_families=(
            "angle_next = angle + dt * angular_velocity for the first three coordinates",
            "low-confidence sparse action-to-joint-velocity laws",
        ),
        forbidden_law_families=(
            "do not invent global x/y position updates because those coordinates are absent",
            "do not use a half-position/half-velocity split",
        ),
        reward_semantics="forward swimming reward minus action-control penalty",
        termination_semantics="does not naturally terminate before truncation",
    ),
    "Hopper-v5": MujocoTaskSpec(
        env_id="Hopper-v5",
        state_dimensions=(
            DimensionSpec(0, "torso_height", "qpos", "vertical torso height", "height_next = height + dt * vertical_velocity", constraints="termination-sensitive", related=("state[6]",)),
            DimensionSpec(1, "torso_angle", "qpos_angle", "torso pitch angle", "angle_next = angle + dt * angular_velocity", constraints="termination-sensitive", related=("state[7]",)),
            DimensionSpec(2, "thigh_angle", "qpos_angle", "thigh joint angle", "angle_next = angle + dt * angular_velocity", related=("state[8]", "action[0]")),
            DimensionSpec(3, "leg_angle", "qpos_angle", "leg joint angle", "angle_next = angle + dt * angular_velocity", related=("state[9]", "action[1]")),
            DimensionSpec(4, "foot_angle", "qpos_angle", "foot joint angle", "angle_next = angle + dt * angular_velocity", related=("state[10]", "action[2]")),
            DimensionSpec(5, "root_x_velocity", "qvel", "forward root velocity", "reward-relevant velocity", related=("action[0]", "action[1]", "action[2]")),
            DimensionSpec(6, "root_z_velocity", "qvel", "vertical root velocity", "height dynamics", related=("state[0]",)),
            DimensionSpec(7, "torso_angular_velocity", "qvel", "torso angular velocity", "balance dynamics", related=("state[1]",)),
            DimensionSpec(8, "thigh_angular_velocity", "qvel", "thigh joint angular velocity", "torque-driven acceleration", related=("state[2]", "action[0]")),
            DimensionSpec(9, "leg_angular_velocity", "qvel", "leg joint angular velocity", "torque-driven acceleration", related=("state[3]", "action[1]")),
            DimensionSpec(10, "foot_angular_velocity", "qvel", "foot joint angular velocity", "torque-driven acceleration", related=("state[4]", "action[2]")),
        ),
        action_dimensions=(
            ActionSpec(0, "thigh_torque", "torque", ("state[8]", "state[5]"), "local joint torque"),
            ActionSpec(1, "leg_torque", "torque", ("state[9]", "state[5]"), "local joint torque"),
            ActionSpec(2, "foot_torque", "torque", ("state[10]", "state[5]"), "local joint torque"),
        ),
        known_structure=(
            "Root x position is excluded; root x velocity is still observed and reward-relevant.",
            "The first five coordinates are qpos after excluding global x; the last six are qvel.",
        ),
        safe_law_families=(
            "qpos_next = qpos + dt * matching_qvel for state[0:5]",
            "very sparse torque-to-matching-joint-velocity laws",
        ),
        forbidden_law_families=(
            "do not invent missing root x position",
            "do not overwrite termination-sensitive height/angle with broad learned constants",
        ),
        reward_semantics="forward velocity and alive bonus minus control penalty",
        termination_semantics="terminates when height, angle, or state validity leaves healthy ranges",
    ),
    "Walker2d-v5": MujocoTaskSpec(
        env_id="Walker2d-v5",
        state_dimensions=(
            DimensionSpec(0, "torso_height", "qpos", "vertical torso height", "height_next = height + dt * vertical_velocity", constraints="termination-sensitive", related=("state[9]",)),
            DimensionSpec(1, "torso_angle", "qpos_angle", "torso pitch angle", "angle_next = angle + dt * angular_velocity", constraints="termination-sensitive", related=("state[10]",)),
            DimensionSpec(2, "right_thigh_angle", "qpos_angle", "right thigh joint angle", "angle_next = angle + dt * angular_velocity", related=("state[11]", "action[0]")),
            DimensionSpec(3, "right_leg_angle", "qpos_angle", "right leg joint angle", "angle_next = angle + dt * angular_velocity", related=("state[12]", "action[1]")),
            DimensionSpec(4, "right_foot_angle", "qpos_angle", "right foot joint angle", "angle_next = angle + dt * angular_velocity", related=("state[13]", "action[2]")),
            DimensionSpec(5, "left_thigh_angle", "qpos_angle", "left thigh joint angle", "angle_next = angle + dt * angular_velocity", related=("state[14]", "action[3]")),
            DimensionSpec(6, "left_leg_angle", "qpos_angle", "left leg joint angle", "angle_next = angle + dt * angular_velocity", related=("state[15]", "action[4]")),
            DimensionSpec(7, "left_foot_angle", "qpos_angle", "left foot joint angle", "angle_next = angle + dt * angular_velocity", related=("state[16]", "action[5]")),
            DimensionSpec(8, "root_x_velocity", "qvel", "forward root velocity", "reward-relevant velocity", related=("action[0]", "action[1]", "action[2]", "action[3]", "action[4]", "action[5]")),
            DimensionSpec(9, "root_z_velocity", "qvel", "vertical root velocity", "height dynamics", related=("state[0]",)),
            DimensionSpec(10, "torso_angular_velocity", "qvel", "torso angular velocity", "balance dynamics", related=("state[1]",)),
            DimensionSpec(11, "right_thigh_angular_velocity", "qvel", "right thigh angular velocity", "torque-driven acceleration", related=("state[2]", "action[0]")),
            DimensionSpec(12, "right_leg_angular_velocity", "qvel", "right leg angular velocity", "torque-driven acceleration", related=("state[3]", "action[1]")),
            DimensionSpec(13, "right_foot_angular_velocity", "qvel", "right foot angular velocity", "torque-driven acceleration", related=("state[4]", "action[2]")),
            DimensionSpec(14, "left_thigh_angular_velocity", "qvel", "left thigh angular velocity", "torque-driven acceleration", related=("state[5]", "action[3]")),
            DimensionSpec(15, "left_leg_angular_velocity", "qvel", "left leg angular velocity", "torque-driven acceleration", related=("state[6]", "action[4]")),
            DimensionSpec(16, "left_foot_angular_velocity", "qvel", "left foot angular velocity", "torque-driven acceleration", related=("state[7]", "action[5]")),
        ),
        action_dimensions=(
            ActionSpec(0, "right_thigh_torque", "torque", ("state[11]",), "local joint torque"),
            ActionSpec(1, "right_leg_torque", "torque", ("state[12]",), "local joint torque"),
            ActionSpec(2, "right_foot_torque", "torque", ("state[13]",), "local joint torque"),
            ActionSpec(3, "left_thigh_torque", "torque", ("state[14]",), "local joint torque"),
            ActionSpec(4, "left_leg_torque", "torque", ("state[15]",), "local joint torque"),
            ActionSpec(5, "left_foot_torque", "torque", ("state[16]",), "local joint torque"),
        ),
        known_structure=(
            "Root x position is excluded; root x velocity remains observed.",
            "The first eight coordinates are qpos after excluding global x; the last nine are qvel.",
        ),
        safe_law_families=(
            "qpos_next = qpos + dt * matching_qvel for state[0:8]",
            "low-confidence sparse torque-to-matching-joint-velocity laws",
        ),
        forbidden_law_families=(
            "do not invent missing root x position",
            "do not overwrite healthy height/angle with broad constants",
        ),
        reward_semantics="forward walking velocity and alive bonus minus control penalty",
        termination_semantics="terminates when height, angle, or state validity leaves healthy ranges",
    ),
    "HalfCheetah-v5": MujocoTaskSpec(
        env_id="HalfCheetah-v5",
        state_dimensions=(
            DimensionSpec(0, "root_z_position", "qpos", "root vertical position", "z_next = z + dt * z_velocity", related=("state[9]",)),
            DimensionSpec(1, "root_pitch", "qpos_angle", "root pitch angle", "angle_next = angle + dt * angular_velocity", related=("state[10]",)),
            DimensionSpec(2, "back_thigh_angle", "qpos_angle", "back thigh joint angle", "angle_next = angle + dt * angular_velocity", related=("state[11]", "action[0]")),
            DimensionSpec(3, "back_shin_angle", "qpos_angle", "back shin joint angle", "angle_next = angle + dt * angular_velocity", related=("state[12]", "action[1]")),
            DimensionSpec(4, "back_foot_angle", "qpos_angle", "back foot joint angle", "angle_next = angle + dt * angular_velocity", related=("state[13]", "action[2]")),
            DimensionSpec(5, "front_thigh_angle", "qpos_angle", "front thigh joint angle", "angle_next = angle + dt * angular_velocity", related=("state[14]", "action[3]")),
            DimensionSpec(6, "front_shin_angle", "qpos_angle", "front shin joint angle", "angle_next = angle + dt * angular_velocity", related=("state[15]", "action[4]")),
            DimensionSpec(7, "front_foot_angle", "qpos_angle", "front foot joint angle", "angle_next = angle + dt * angular_velocity", related=("state[16]", "action[5]")),
            DimensionSpec(8, "root_x_velocity", "qvel", "forward root velocity", "reward-relevant velocity", related=("action[0]", "action[1]", "action[2]", "action[3]", "action[4]", "action[5]")),
            DimensionSpec(9, "root_z_velocity", "qvel", "vertical root velocity", "root vertical dynamics", related=("state[0]",)),
            DimensionSpec(10, "root_angular_velocity", "qvel", "root angular velocity", "root pitch dynamics", related=("state[1]",)),
            DimensionSpec(11, "back_thigh_angular_velocity", "qvel", "back thigh angular velocity", "torque-driven acceleration", related=("state[2]", "action[0]")),
            DimensionSpec(12, "back_shin_angular_velocity", "qvel", "back shin angular velocity", "torque-driven acceleration", related=("state[3]", "action[1]")),
            DimensionSpec(13, "back_foot_angular_velocity", "qvel", "back foot angular velocity", "torque-driven acceleration", related=("state[4]", "action[2]")),
            DimensionSpec(14, "front_thigh_angular_velocity", "qvel", "front thigh angular velocity", "torque-driven acceleration", related=("state[5]", "action[3]")),
            DimensionSpec(15, "front_shin_angular_velocity", "qvel", "front shin angular velocity", "torque-driven acceleration", related=("state[6]", "action[4]")),
            DimensionSpec(16, "front_foot_angular_velocity", "qvel", "front foot angular velocity", "torque-driven acceleration", related=("state[7]", "action[5]")),
        ),
        action_dimensions=(
            ActionSpec(0, "back_thigh_torque", "torque", ("state[11]",), "local joint torque"),
            ActionSpec(1, "back_shin_torque", "torque", ("state[12]",), "local joint torque"),
            ActionSpec(2, "back_foot_torque", "torque", ("state[13]",), "local joint torque"),
            ActionSpec(3, "front_thigh_torque", "torque", ("state[14]",), "local joint torque"),
            ActionSpec(4, "front_shin_torque", "torque", ("state[15]",), "local joint torque"),
            ActionSpec(5, "front_foot_torque", "torque", ("state[16]",), "local joint torque"),
        ),
        known_structure=(
            "Global x position is excluded; root x velocity is observed and reward-relevant.",
            "The first eight coordinates are qpos after excluding global x; the last nine are qvel.",
            "The environment normally ends only by time truncation.",
        ),
        safe_law_families=(
            "qpos_next = qpos + dt * matching_qvel for state[0:8]",
            "low-confidence sparse torque-to-matching-joint-velocity laws",
        ),
        forbidden_law_families=(
            "do not invent missing root x position",
            "do not write broad velocity laws that ignore contact and actuator coupling",
        ),
        reward_semantics="forward velocity minus control penalty",
        termination_semantics="does not naturally terminate before truncation",
    ),
}
