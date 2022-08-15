print("Hello World")
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
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.evaluation import evaluate_policy
#################
import torch
import torchvision
#time.clock = time.time
import os
from gym.envs.box2d.car_racing import CarRacing #change to car_racing_kc for custom car racing environment
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)#seed for repeatable results

env = CarRacing()
#print("hello world")

env.seed(RANDOM_SEED)#seed for repeatable results
np.random.seed(RANDOM_SEED)

print(env.observation_space.shape)

# print("Action Space: {}".format(env.action_space))
# print("State space: {}".format(env.observation_space))

# An episode a full game
train_episodes = 300
test_episodes = 100


def agent(state_shape, action_shape): #Build model DQN
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    learning_rate = 0.01
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=state_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(216, activation='relu'))
    model.add(Dense(action_shape, activation=None))
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate, epsilon=1e-7))



    # init = tf.keras.initializers.HeUniform()
    # model = keras.Sequential()
    # model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
    # model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    # #model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    # model.add(Dense(216, activation='relu', kernel_initializer=init))
    # model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    # model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
    #               metrics=['accuracy'])
    return model

#Function to get q values
def get_qs(model, state, step):
    return model.predict(state.reshape([1, state.shape[0], state.shape[1], state.shape[2]]))[0]

#Training function
def train(env, replay_memory, model, target_model, done):
    learning_rate = 0.1  # Learning rate
    discount_factor = 0.618

    MIN_REPLAY_SIZE = 10000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 34 * 2 #size on which the model trains
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        #print(action)
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        print("maxfutureq")
        print(observation, action, reward, new_observation, done)
        #max_index = list(action).index(np.argmax(action))
        #print(max_index)
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)


def main():
    epsilon = 1  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1  # You can't explore more than 100% of the time
    min_epsilon = 0.01  # At a minimum, we'll always explore 1% of the time
    decay = 0.01

    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    print("env action space")
    print(env.action_space.n)
    model = agent(env.observation_space.shape, env.action_space.n)
    # Target Model (updated every 100 steps)
    #    target_model = agent(env.observation_space.shape, env.action_space.n)
    target_model = agent(env.observation_space.shape, env.action_space.n)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=50_000)

    target_update_counter = 0

    # X = states, y = actions
    X = []
    y = []

    steps_to_update_target_model = 0

    for episode in range(train_episodes):
        total_training_rewards = 0
        observation = env.reset()
        done = False
        while not done:
            steps_to_update_target_model += 1
            if True:
                env.render()

            random_number = np.random.rand()
            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number <= epsilon:
                # Explore
                action = env.action_space.sample()
            else:
                # Exploit best known action
                # model dims are (batch, env.observation_space.n)
                encoded = observation
                encoded_reshaped = encoded.reshape([1, encoded.shape[0], encoded.shape[1], encoded.shape[2]])
                print("Re-shaped",encoded_reshaped.shape)
                predicted = model.predict(encoded_reshaped).flatten()
                print("I am predicted: ", predicted)
                action = np.argmax(predicted)
                print("I am best action")
                print(action)

            #print(action)
            #print(len(env.step(action)))
            new_observation, reward, done, info = env.step(action)
            #print(reward)
            replay_memory.append([observation, action, reward, new_observation, done])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                train(env, replay_memory, model, target_model, done)

            observation = new_observation
            total_training_rewards += reward
            #print(total_training_rewards)
            if abs(total_training_rewards) > 125:
                done = True
            if done:
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(
                    total_training_rewards, episode, reward))
                total_training_rewards += 1

                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
    env.close()


if __name__ == '__main__':
    main()