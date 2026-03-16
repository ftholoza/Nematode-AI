import math
import numpy as np

from brain import Brain


WORLD_XMIN, WORLD_XMAX = 0.0, 10.0
WORLD_YMIN, WORLD_YMAX = 0.0, 10.0

SENSOR_OFFSET_ANGLE = math.radians(35.0)
SENSOR_RANGE = 10.0
NEAR_THRESHOLD = 0.45

TURN_GAIN = 0.22
FORWARD_GAIN = 0.20
BASE_FORWARD_SPEED = 0.02
BRAKE_GAIN = 0.18

SEARCH_THRESHOLD = 0.25
SEARCH_DRIVE = 0.25

MIN_FORWARD_SPEED = 0.015
TARGET_RADIUS = 0.70


def angle_diff(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def sensor_response(
    sensor_origin: np.ndarray,
    sensor_angle: float,
    target_pos: np.ndarray,
    sensor_range: float,
) -> tuple[float, float]:
    to_target = target_pos - sensor_origin
    dist = np.linalg.norm(to_target)

    if dist < 1e-9:
        return 0.0, 1.0

    if dist > sensor_range:
        return 0.0, 0.0

    target_angle = math.atan2(to_target[1], to_target[0])
    diff = angle_diff(target_angle, sensor_angle)

    direction_factor = max(0.0, (math.cos(diff) + 1.0) / 2.0)

    proximity = 1.0 - (dist / sensor_range)
    distance_factor = dist / sensor_range

    near_raw = proximity
    far_raw = distance_factor

    near_value = direction_factor * max(
        0.0, (near_raw - (1.0 - NEAR_THRESHOLD)) / NEAR_THRESHOLD
    )
    far_value = direction_factor * max(
        0.0, (far_raw - NEAR_THRESHOLD) / (1.0 - NEAR_THRESHOLD)
    )

    near_value = float(np.clip(near_value, 0.0, 1.0))
    far_value = float(np.clip(far_value, 0.0, 1.0))

    return far_value, near_value


def compute_sensors(
    agent_pos: np.ndarray,
    agent_angle: float,
    target_pos: np.ndarray,
) -> dict[str, float]:
    left_sensor_angle = agent_angle + SENSOR_OFFSET_ANGLE
    right_sensor_angle = agent_angle - SENSOR_OFFSET_ANGLE

    left_far, left_near = sensor_response(
        sensor_origin=agent_pos,
        sensor_angle=left_sensor_angle,
        target_pos=target_pos,
        sensor_range=SENSOR_RANGE,
    )

    right_far, right_near = sensor_response(
        sensor_origin=agent_pos,
        sensor_angle=right_sensor_angle,
        target_pos=target_pos,
        sensor_range=SENSOR_RANGE,
    )

    return {
        "S_left_far": left_far,
        "S_left_near": left_near,
        "S_right_far": right_far,
        "S_right_near": right_near,
    }


def update_agent(
    pos: np.ndarray,
    angle: float,
    motor_left: float,
    motor_right: float,
    motor_forward: float,
    motor_brake: float,
    target_pos: np.ndarray,
) -> tuple[np.ndarray, float]:
    distance = np.linalg.norm(target_pos - pos)

    turn_damping = min(1.0, distance / 3.0)
    turn = TURN_GAIN * (motor_left - motor_right) * turn_damping

    forward = BASE_FORWARD_SPEED + FORWARD_GAIN * max(0.0, motor_forward)
    forward -= BRAKE_GAIN * max(0.0, motor_brake)
    forward = max(MIN_FORWARD_SPEED, forward)

    if distance < 1.2:
        forward *= distance / 1.2

    new_angle = angle + turn

    dx = forward * math.cos(new_angle)
    dy = forward * math.sin(new_angle)

    new_pos = pos + np.array([dx, dy], dtype=float)
    new_pos[0] = np.clip(new_pos[0], WORLD_XMIN, WORLD_XMAX)
    new_pos[1] = np.clip(new_pos[1], WORLD_YMIN, WORLD_YMAX)

    return new_pos, new_angle


def run_episode(
    brain: Brain,
    target_pos: np.ndarray,
    initial_agent_pos: np.ndarray,
    initial_agent_angle: float,
    max_steps: int = 300,
) -> dict:
    brain.reset()

    agent_pos = initial_agent_pos.astype(float).copy()
    agent_angle = float(initial_agent_angle)

    initial_distance = float(np.linalg.norm(target_pos - agent_pos))
    best_distance = initial_distance

    reached = False
    reached_step = None

    for step in range(max_steps):
        distance_to_target = float(np.linalg.norm(target_pos - agent_pos))

        if distance_to_target <= TARGET_RADIUS:
            reached = True
            reached_step = step
            break

        sensor_values = compute_sensors(agent_pos, agent_angle, target_pos)

        sensor_sum = (
            sensor_values["S_left_far"]
            + sensor_values["S_left_near"]
            + sensor_values["S_right_far"]
            + sensor_values["S_right_near"]
        )

        sensory = brain.build_sensory_vector(sensor_values)

        if sensor_sum < SEARCH_THRESHOLD:
            sensory[brain.get_neuron_id("I_search")] = SEARCH_DRIVE

        brain.step(sensory)

        motor_left = brain.get_activity_by_name("M_left")
        motor_right = brain.get_activity_by_name("M_right")
        motor_forward = brain.get_activity_by_name("M_forward")
        motor_brake = brain.get_activity_by_name("M_brake")

        agent_pos, agent_angle = update_agent(
            agent_pos,
            agent_angle,
            motor_left,
            motor_right,
            motor_forward,
            motor_brake,
            target_pos,
        )

        best_distance = min(best_distance, float(np.linalg.norm(target_pos - agent_pos)))

    final_distance = float(np.linalg.norm(target_pos - agent_pos))

    # Simple score
    score = (initial_distance - final_distance) * 10.0
    score += (initial_distance - best_distance) * 5.0

    if reached:
        score += 100.0
        score -= 0.1 * reached_step
    else:
        score -= 0.05 * max_steps

    return {
        "reached": reached,
        "reached_step": reached_step,
        "initial_distance": initial_distance,
        "best_distance": best_distance,
        "final_distance": final_distance,
        "score": score,
        "final_agent_pos": agent_pos.copy(),
        "final_agent_angle": agent_angle,
    }