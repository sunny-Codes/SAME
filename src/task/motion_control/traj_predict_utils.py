""" 
original code: 
https://github.com/orangeduck/Motion-Matching/blob/main/controller.cpp

copied and converted from C++ to python
"""

from fairmotion.ops import conversions
import numpy as np
from fairmotion.utils import spring_utils
from utils.data_utils import safe_normalize_to_one


def orbit_camera_update_azimuth(
    azimuth,  # const float
    gamepadstick_right,  # const vec3
    desired_strafe,  # const bool
    dt,  # const float
):
    gamepadaxis = np.zeros(3) if desired_strafe else gamepadstick_right
    return azimuth + 2.0 * dt * -gamepadaxis[0]


def desired_rotation_update(
    desired_rotation,  # const quat
    gamepadstick_left,  # const vec3
    gamepadstick_right,  # const vec3
    camera_azimuth,  # const float
    desired_strafe,  # const bool
    desired_velocity,  # const vec3
):
    desired_rotation_curr = desired_rotation

    # If strafe is active then desired direction is coming from right
    # stick as long as that stick is being used, otherwise we assume
    # forward facing
    if desired_strafe:
        desired_direction = conversions.A2R(
            camera_azimuth * np.array([0, 1, 0])
        ) @ np.array([0, 0, -1])

        if np.linalg.norm(gamepadstick_right) > 0.01:
            desired_direction = conversions.A2R(
                camera_azimuth * np.array([0, 1, 0])
            ) @ safe_normalize_to_one(gamepadstick_right)

        return conversions.A2R(
            np.arctan2(desired_direction[0], desired_direction[2]) * np.array([0, 1, 0])
        )

    # If strafe is not active the desired direction comes from the left
    # stick as long as that stick is being used
    elif np.linalg.norm(gamepadstick_left) > 0.01:
        desired_direction = safe_normalize_to_one(desired_velocity)
        return conversions.A2R(
            np.arctan2(desired_direction[0], desired_direction[2]) * np.array([0, 1, 0])
        )

    # Otherwise desired direction remains the same
    else:
        return desired_rotation_curr


def desired_velocity_update(
    gamepadstick_left,  # const vec3
    camera_azimuth,  # const float
    simulation_rotation,  # const quat
    fwrd_speed,  # const float
    side_speed,  # const float
    back_speed,  # const float
):
    # Find stick position in world space by rotating using camera azimuth
    global_stick_direction = (
        conversions.A2R(camera_azimuth * np.array([0, 1, 0])) @ gamepadstick_left
    )

    # Find stick position local to current facing direction
    local_stick_direction = np.linalg.inv(simulation_rotation) @ global_stick_direction

    # Scale stick by forward, sideways and backwards speeds
    if local_stick_direction[2] > 0.0:
        local_desired_velocity = (
            np.array([side_speed, 0.0, fwrd_speed]) * local_stick_direction
        )
    else:
        local_desired_velocity = (
            np.array([side_speed, 0.0, back_speed]) * local_stick_direction
        )

    # Re-orientate into the world space
    return simulation_rotation @ local_desired_velocity


# # Predict desired rotations given the estimated future
# # camera rotation and other parameters
def traj_desired_rotations_predict(
    desired_rotations,  # slice1d<quat>
    desired_velocities,  # const slice1d<vec3>
    desired_rotation,  # const quat
    camera_azimuth,  # const float
    gamepadstick_left,  # const vec3
    gamepadstick_right,  # const vec3
    desired_strafe,  # const bool
    dt,  # const float
):
    # in-place function
    # changes only desired_rotations (other are constant)
    desired_rotations[0] = desired_rotation

    for i in range(1, len(desired_rotations)):
        desired_rotations[i] = desired_rotation_update(
            desired_rotations[i - 1],
            gamepadstick_left,
            gamepadstick_right,
            orbit_camera_update_azimuth(
                camera_azimuth, gamepadstick_right, desired_strafe, i * dt
            ),
            desired_strafe,
            desired_velocities[i],
        )


def traj_rotations_predict(
    rotations,  # slice1d<quat>
    angular_velocities,  # slice1d<vec3>
    rotation,  # const quat
    angular_velocity,  # const vec3
    desired_rotations,  # const slice1d<quat>
    halflife,  # const float
    dt,  # const float
):
    # in-place function
    # changes only rotations & angular_velocities (other are constant)
    for i in range(len(rotations)):
        rotations[i] = rotation
    for i in range(len(angular_velocities)):
        angular_velocities[i] = angular_velocity

    for i in range(1, len(rotations)):
        rotations[i], angular_velocities[i] = (
            spring_utils.simple_spring_damper_implicit_quat(
                rotations[i],
                angular_velocities[i],
                desired_rotations[i],
                halflife,
                i * dt,
            )
        )


# Predict what the desired velocity will be in the
# future. Here we need to use the future trajectory
# rotation as well as predicted future camera
# position to find an accurate desired velocity in
# the world space
def traj_desired_velocities_predict(
    desired_velocities,  # slice1d<vec3>
    trajectory_rotations,  # const slice1d<quat>
    desired_velocity,  # const vec3
    camera_azimuth,  # const float
    gamepadstick_left,  # const vec3
    gamepadstick_right,  # const vec3
    desired_strafe,  # const bool
    fwrd_speed,  # const float
    side_speed,  # const float
    back_speed,  # const float
    dt,  # const float
):
    desired_velocities[0] = desired_velocity

    for i in range(1, len(desired_velocities)):
        desired_velocities[i] = desired_velocity_update(
            gamepadstick_left,
            orbit_camera_update_azimuth(
                camera_azimuth, gamepadstick_right, desired_strafe, i * dt
            ),
            trajectory_rotations[i],
            fwrd_speed,
            side_speed,
            back_speed,
        )


def traj_positions_predict(
    positions,  # slice1d<vec3>
    velocities,  # slice1d<vec3>
    accelerations,  # slice1d<vec3>
    position,  # const vec3
    velocity,  # const vec3
    acceleration,  # const vec3
    desired_velocities,  # const slice1d<vec3>
    halflife,  # const float
    dt,  # const float
    # obstacles_positions, # const slice1d<vec3>
    # obstacles_scales # const slice1d<vec3>
):
    positions[0] = position
    velocities[0] = velocity
    accelerations[0] = acceleration

    for i in range(1, len(positions)):
        positions[i] = positions[i - 1]
        velocities[i] = velocities[i - 1]
        accelerations[i] = accelerations[i - 1]

        positions[i], velocities[i], accelerations[i] = (
            spring_utils.simple_spring_damper_implicit_a(
                positions[i],
                velocities[i],
                accelerations[i],
                desired_velocities[i],
                halflife,
                dt,
                # obstacles_positions,
                # obstacles_scales.
            )
        )
