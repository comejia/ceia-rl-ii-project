import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

# observaciones
#
# 7 posiciones
# 7 velocidades
# 3 ee
# 3 posición target
# 3 error EE-target
# -----------------
# 23 valores


class RobotArmEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()

        self.max_steps = 200
        self.steps = 0
        self.render_mode = render
        self.previous_distance = None

        if render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # 7 articulaciones
        self.n_joints = 7

        # acciones continuas
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(self.n_joints,), dtype=np.float32
        )

        # observaciones
        obs_size = 23

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.max_steps = 200

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0

        p.resetSimulation()

        p.setGravity(0, 0, -9.81)

        self.plane = p.loadURDF("plane.urdf")

        self.robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

        # self.target_pos = np.random.uniform(
        #    low=[0.4, -0.3, 0.2],
        #    high=[0.7, 0.3, 0.6]
        # )
        self.target_pos = np.random.uniform(low=[0.5, -0.1, 0.4], high=[0.6, 0.1, 0.5])

        self.target = p.loadURDF("sphere_small.urdf", basePosition=self.target_pos)

        self.previous_distance = self._distance_to_target()

        obs = self._get_obs()

        return obs, {}

    def _get_obs(self):
        obs = []

        for i in range(self.n_joints):
            joint_state = p.getJointState(self.robot, i)

            position = joint_state[0]
            velocity = joint_state[1]

            obs.append(position)
            obs.append(velocity)

        link_state = p.getLinkState(self.robot, 6)

        ee_pos = link_state[0]

        obs.extend(ee_pos)

        obs.extend(self.target_pos)

        # vector desde el efector final hacia el target
        error = np.array(self.target_pos) - np.array(ee_pos)

        obs.extend(error)

        return np.array(obs, dtype=np.float32)

    def _get_joint_positions(self):
        positions = []

        for i in range(self.n_joints):
            joint_state = p.getJointState(self.robot, i)

            positions.append(joint_state[0])

        return np.array(positions, dtype=np.float32)

    def _get_end_effector_position(self):
        link_state = p.getLinkState(self.robot, 6)

        return np.array(link_state[0], dtype=np.float32)

    def _distance_to_target(self):
        ee_pos = self._get_end_effector_position()

        return np.linalg.norm(ee_pos - self.target_pos)

    def _compute_reward_distance(self):
        """Computa el reward basado en la distancia absoluta."""

        distance = self._distance_to_target()

        return -distance

    def _compute_reward_progress(self):
        """Computa el reward premiando el progreso."""

        current_distance = self._distance_to_target()

        reward = self.previous_distance - current_distance

        self.previous_distance = current_distance

        return reward

    def _compute_reward_progress_shaping(self):
        current_distance = self._distance_to_target()

        progress = self.previous_distance - current_distance

        # bonus suave cuando está cerca
        if current_distance < 0.2:
            progress += 1.0 * (0.2 - current_distance)

        self.previous_distance = current_distance

        return progress

    def apply_action(self, action):
        current_positions = self._get_joint_positions()

        target_positions = current_positions + action

        for joint in range(self.n_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(target_positions[joint]),
                force=500,
            )

        # p.stepSimulation()
        for _ in range(5):
            p.stepSimulation()

    def step(self, action):
        self.apply_action(action)

        self.steps += 1

        obs = self._get_obs()

        reward = self._compute_reward_progress_shaping()

        distance = self._distance_to_target()

        terminated = distance < 0.05

        if terminated:
            reward += 100

        truncated = self.steps >= self.max_steps

        info = {}

        return (obs, reward, terminated, truncated, info)

    def close(self):
        import pybullet as p

        if p.isConnected():
            p.disconnect()
