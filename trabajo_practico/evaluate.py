import numpy as np
from stable_baselines3 import PPO
from robot_arm_env import RobotArmEnv


RENDER = False
EPISODES = 100
MODEL = "ppo_robot_arm_v2_5cm"

model = PPO.load(MODEL)

env = RobotArmEnv(render=RENDER)

successes = 0
rewards = []
distances = []
steps_success = []


for episode in range(EPISODES):
    obs, info = env.reset()

    terminated = False
    truncated = False

    episode_reward = 0
    steps = 0

    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        steps += 1

    final_distance = env._distance_to_target()

    rewards.append(episode_reward)

    if final_distance < 0.05:
        successes += 1
        steps_success.append(steps)

    distances.append(final_distance)

    print(
        f"Episode {episode + 1}: "
        f"reward={episode_reward:.3f}, "
        f"distance={final_distance:.3f}, "
        f"success={terminated}"
    )


success_rate = successes / EPISODES

print("\n======================")
print(f"Success rate: {success_rate * 100:.2f}%")
print(f"Average reward: {np.mean(rewards):.3f}")
print(f"Average final distance: {np.mean(distances):.3f}")
print(f"Average steps to success: {np.mean(steps_success):.3f}")
print(f"Best distance: {np.min(distances):.3f}")
print("======================")

env.close()
