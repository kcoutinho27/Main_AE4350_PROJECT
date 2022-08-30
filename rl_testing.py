import gym
import tensorflow as tf
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

from tensorflow import keras
from collections import deque
import time
import random
from tensorflow import keras
#from DQN_agent_kc import agent, qvals, train_agent
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
#################
import torch
import torchvision
time.clock = time.time
import os
from gym.envs.box2d.car_racing_mike import CarRacing #change to car_racing_kc for custom car racing environment
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

#If you want to render in human mode, initialize the environment in this way: gym.make('EnvName', render_mode='human') and don't call the render method.

env = CarRacing()
env.reset()
new_step_api=True

# #################
# #applying algorithm
# env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])#use dummy vec as wrapper for stacking frames
#
log_path = os.path.join('Training', 'Logs')
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_path)#ppo algorithm used from stable baselines

model.learn(total_timesteps=40000)#choose steps to train, otherwise car racing environment trains until 900 points
#reward, this may take too long
#del model
ppo_path = os.path.join('Training', 'Saved Models', 'PPO_Driving_model50k')
#del model

model = PPO.load(ppo_path, env)
model.save(ppo_path)

evaluate_policy(model, env, n_eval_episodes=1, render=True)
env.close()


#Test environment with random action, can't take turns
# episodes = 2
# for episode in range(1, episodes + 1):
#     state = env.reset()
#     done = False
#     score = 0
#
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score += reward
#     print('Episode:{} Score:{}'.format(episode, score))
# env.close()

