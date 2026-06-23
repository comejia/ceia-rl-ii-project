from stable_baselines3 import PPO

from robot_arm_env import RobotArmEnv

from stable_baselines3.common.monitor import Monitor

env = Monitor(
    RobotArmEnv(render=False),
    filename="monitor.csv"
)


# model = PPO(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     device="cpu",
#     tensorboard_log="./logs"
# )


# Transfer Learning
model = PPO.load(
    "ppo_robot_arm_v1_5cm",
    env=env,
    verbose=1,
    device="cpu",
    tensorboard_log="./logs"
)

model.learn(
    total_timesteps=400_000,
    tb_log_name="fine_tune_5cm"
)


# model.learn(
#     total_timesteps=200_000,
#     tb_log_name="rew_distance_progress_shaping_error"
# )

print("Entrenamiento finalizado")

model.save("ppo_robot_arm_v2_5cm")

print("Modelo guardado")