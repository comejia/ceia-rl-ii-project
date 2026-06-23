from stable_baselines3 import PPO

from robot_arm_env import RobotArmEnv

from stable_baselines3.common.monitor import Monitor

env = Monitor(RobotArmEnv(render=False), filename="monitor.csv")


# Train model
model = PPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log="./logs")

# # Transfer Learning
# model = PPO.load(
#     "<model_saved>",
#     env=env,
#     verbose=1,
#     device="cpu",
#     tensorboard_log="./logs"
# )

model.learn(total_timesteps=400_000, tb_log_name="ppo_robot_arm_v2_5cm")


print("Entrenamiento finalizado")

model.save("ppo_robot_arm_v2_5cm")

print("Modelo guardado")
