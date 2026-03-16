import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from brain import Brain


# =====================================
# Config
# =====================================
WORLD_XMIN, WORLD_XMAX = 0.0, 10.0
WORLD_YMIN, WORLD_YMAX = 0.0, 10.0

TARGET_POS = np.array([7.5, 7.0], dtype=float)

INITIAL_AGENT_POS = np.array([2.0, 2.0], dtype=float)
INITIAL_AGENT_ANGLE = math.radians(20.0)

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

N_FRAMES = 500
FRAME_INTERVAL_MS = 120

OUTPUT_DIR = "visualize"
OUTPUT_MP4 = os.path.join(OUTPUT_DIR, "environment.mp4")


# =====================================
# Brain
# =====================================
brain = Brain(alpha=0.2)


# =====================================
# Agent state
# =====================================
agent_pos = INITIAL_AGENT_POS.copy()
agent_angle = INITIAL_AGENT_ANGLE
trajectory = [agent_pos.copy()]
target_reached = False
target_reached_frame = None


# =====================================
# Helpers
# =====================================
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
    """
    Returns (far_value, near_value) in [0,1].
    """
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


def compute_sensors(agent_pos: np.ndarray, agent_angle: float) -> dict[str, float]:
    left_sensor_angle = agent_angle + SENSOR_OFFSET_ANGLE
    right_sensor_angle = agent_angle - SENSOR_OFFSET_ANGLE

    left_far, left_near = sensor_response(
        sensor_origin=agent_pos,
        sensor_angle=left_sensor_angle,
        target_pos=TARGET_POS,
        sensor_range=SENSOR_RANGE,
    )

    right_far, right_near = sensor_response(
        sensor_origin=agent_pos,
        sensor_angle=right_sensor_angle,
        target_pos=TARGET_POS,
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

    # If turning seems inverted, swap motor_left and motor_right here
    turn = TURN_GAIN * (motor_left - motor_right) * turn_damping

    forward = BASE_FORWARD_SPEED + FORWARD_GAIN * max(0.0, motor_forward)
    forward -= BRAKE_GAIN * max(0.0, motor_brake)

    # Prevent full immobilization
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


def draw_agent(ax, pos: np.ndarray, angle: float) -> None:
    ax.scatter(pos[0], pos[1], s=120)

    hx = pos[0] + 0.55 * math.cos(angle)
    hy = pos[1] + 0.55 * math.sin(angle)
    ax.plot([pos[0], hx], [pos[1], hy], linewidth=2)

    left_angle = angle + SENSOR_OFFSET_ANGLE
    right_angle = angle - SENSOR_OFFSET_ANGLE

    lx = pos[0] + 0.90 * math.cos(left_angle)
    ly = pos[1] + 0.90 * math.sin(left_angle)
    rx = pos[0] + 0.90 * math.cos(right_angle)
    ry = pos[1] + 0.90 * math.sin(right_angle)

    ax.plot([pos[0], lx], [pos[1], ly], linestyle="--", linewidth=1)
    ax.plot([pos[0], rx], [pos[1], ry], linestyle="--", linewidth=1)


# =====================================
# Figure
# =====================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, (ax_world, ax_info) = plt.subplots(
    1,
    2,
    figsize=(12, 6),
    gridspec_kw={"width_ratios": [3.1, 1.3]},
)

ax_info.axis("off")


# =====================================
# Animation
# =====================================
def animate(frame: int):
    global agent_pos, agent_angle, trajectory, target_reached, target_reached_frame

    distance_to_target = np.linalg.norm(TARGET_POS - agent_pos)

    if not target_reached and distance_to_target <= TARGET_RADIUS:
        target_reached = True
        target_reached_frame = frame
        print(f"Target reached at frame {frame}")

    if not target_reached:
        sensor_values = compute_sensors(agent_pos, agent_angle)

        sensor_sum = (
            sensor_values["S_left_far"]
            + sensor_values["S_left_near"]
            + sensor_values["S_right_far"]
            + sensor_values["S_right_near"]
        )

        sensory = brain.build_sensory_vector(sensor_values)

        # Search drive when signal is weak
        if sensor_sum < SEARCH_THRESHOLD:
            sensory[brain.get_neuron_id("I_search")] = SEARCH_DRIVE

        x, total_input = brain.step(sensory)

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
            TARGET_POS,
        )

        trajectory.append(agent_pos.copy())

    else:
        sensor_values = {
            "S_left_far": 0.0,
            "S_left_near": 0.0,
            "S_right_far": 0.0,
            "S_right_near": 0.0,
        }

        sensory = brain.build_sensory_vector(sensor_values)
        x, total_input = brain.step(sensory)

        motor_left = 0.0
        motor_right = 0.0
        motor_forward = 0.0
        motor_brake = 0.0
        sensor_sum = 0.0

    distance_to_target = np.linalg.norm(TARGET_POS - agent_pos)

    # ----- World panel
    ax_world.clear()
    ax_world.set_xlim(WORLD_XMIN, WORLD_XMAX)
    ax_world.set_ylim(WORLD_YMIN, WORLD_YMAX)
    ax_world.set_aspect("equal")

    if target_reached:
        ax_world.set_title(f"Target reached — t={frame}")
    else:
        ax_world.set_title(f"Agent in 2D environment — t={frame}")

    ax_world.scatter(
        TARGET_POS[0],
        TARGET_POS[1],
        s=180,
        marker="*",
        label="target",
    )

    target_circle = plt.Circle(
        (TARGET_POS[0], TARGET_POS[1]),
        TARGET_RADIUS,
        fill=False,
        linestyle=":",
        linewidth=1,
    )
    ax_world.add_patch(target_circle)

    traj = np.array(trajectory)
    ax_world.plot(traj[:, 0], traj[:, 1], linewidth=1)

    draw_agent(ax_world, agent_pos, agent_angle)

    if target_reached:
        ax_world.scatter(agent_pos[0], agent_pos[1], s=220, marker="o")

    ax_world.legend(loc="upper left")

    # ----- Info panel
    ax_info.clear()
    ax_info.axis("off")

    status = "REACHED" if target_reached else "SEARCHING"

    info_text = f"""
Status
{status}

Sensors
L_far   = {sensor_values["S_left_far"]:.3f}
L_near  = {sensor_values["S_left_near"]:.3f}
R_far   = {sensor_values["S_right_far"]:.3f}
R_near  = {sensor_values["S_right_near"]:.3f}

Sensor sum
sum     = {sensor_sum:.3f}

Interneurons
I_left  = {brain.get_activity_by_name("I_left"):.3f}
I_right = {brain.get_activity_by_name("I_right"):.3f}
I_fwd   = {brain.get_activity_by_name("I_forward"):.3f}
I_srch  = {brain.get_activity_by_name("I_search"):.3f}
I_memL  = {brain.get_activity_by_name("I_memory_left"):.3f}
I_memR  = {brain.get_activity_by_name("I_memory_right"):.3f}

Motors
M_left  = {motor_left:.3f}
M_right = {motor_right:.3f}
M_fwd   = {motor_forward:.3f}
M_brake = {motor_brake:.3f}

Agent
x       = {agent_pos[0]:.3f}
y       = {agent_pos[1]:.3f}
angle   = {agent_angle:.3f}

Target
x       = {TARGET_POS[0]:.3f}
y       = {TARGET_POS[1]:.3f}
dist    = {distance_to_target:.3f}
"""

    if target_reached_frame is not None:
        info_text += f"\nReached at frame\n{target_reached_frame}\n"

    ax_info.text(
        0.0,
        0.98,
        info_text,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
    )


ani = FuncAnimation(
    fig,
    animate,
    frames=N_FRAMES,
    interval=FRAME_INTERVAL_MS,
    repeat=False,
)

ani.save(
    OUTPUT_MP4,
    writer="ffmpeg",
    fps=30,
    bitrate=1800,
)

print(f"Saved animation to {OUTPUT_MP4}")