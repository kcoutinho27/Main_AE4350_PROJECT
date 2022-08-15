import gym
import tensorflow as tf
import numpy as np
from tensorflow import keras
from DQN_agent_kc import agent, qvals, train_agent

from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
#################
import time
import random
import torch
import torchvision
time.clock = time.time
import os
from gym.envs.box2d.car_racing import CarRacing #change to car_racing_kc for custom car racing environment

# gym.envs.register(
#      id='KC_car_racingv0',
#      entry_point='gym.envs.box2d:car_racing',
#      max_episode_steps=150,
#      #kwargs={'size' : 1, 'init_state' : 10.},
# )

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
#env = gym.make('KC_car_racingv0')
env = CarRacing()
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
#print(env.action_space.sample())

#print("Action Space: {}".format(env.action_space))
#print("State space: {}".format(env.observation_space))

episodes = 300
episodes_test = 100

epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
max_epsilon = 1 # You can't explore more than 100% of the time
min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
decay = 0.01

#Two models, Target and main
#Main
model_main = agent(env.observation_space.shape, env.action_space.n)
#Target
target_model = agent(env.observation_space.shape, env.action_space.n)
target_model.set_weights(model.get_weights())

replay_memory = deque(maxlen=50_000)

# X = states, y = actions
    X = []
    y = []


# env.reset()
# new_step_api=True
# env.action_space.sample()
# env.observation_space


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
#env.close()
