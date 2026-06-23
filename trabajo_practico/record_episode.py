from stable_baselines3 import PPO
import time
import imageio
from robot_arm_env import RobotArmEnv
import pybullet as p
import numpy as np

env = RobotArmEnv(render=True)

model = PPO.load(
    "ppo_robot_arm_v2_5cm",
    env=env
)

obs, info = env.reset()

done = False

frames = []


p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=45,
    cameraPitch=-25,
    cameraTargetPosition=[0.4, 0, 0.5]
)


while not done:

    img = p.getCameraImage(
        1024,
        768
    )

    rgb = img[2]

    frame = np.reshape(
        rgb,
        (768, 1024, 4)
    )

    frame = frame[:, :, :3].astype(np.uint8)

    frames.append(frame)


    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    time.sleep(0.05)


print("Target:", env.target_pos)
print("EE:", env._get_end_effector_position())
print("Distance final:", env._distance_to_target())

imageio.mimsave(
    "robot_arm_success.gif",
    frames,
    fps=15
)


input("Presione ENTER para cerrar...")