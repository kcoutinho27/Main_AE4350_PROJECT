#import dependencies
import os #operating system to save
import gym
from stable_baselines3 import PPO #algorithm see baselines documentation
from stable_baselines3.common.vec_env import DummyVecEnv #vectorize env for parralel training
from stable_baselines3.common.evaluation import evaluate_policy #test out how model is running

environment_name = 'CartPole-v0' #mapping to gym environment
env = gym.make(environment_name)
#print(environment_name)

episodes = 1
for episode in range(1, episodes+1):
    state = env.reset() #initial observations
    done = False
    score = 0

    while not done:
        env.render() #can view graphical environment
        action = env.action_space.sample() #generate random action 0 or 1
        n_state, reward, done, info = env.step(action)
        score += reward
    #print('Episode:{} Score:{}'.format(episode, score))
env.close()
#print(env.reset)
#print(env.action_space.sample())

#observation for this environment is cart position, cart velocity, pole angle, pole angular velocity

#directories made
log_path = os.path.join('Training', 'Logs')#monitor training
#print(log_path)

env = gym.make(environment_name)
env = DummyVecEnv([lambda: env]) #wrap environment in dummyvec
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=20000) #how much you want to train, increase for more complex environments

PPO_Path = os.path.join('Training', 'Saved Models', 'PPO_Model_Cartpole')
model.save(PPO_Path)
del model

model = PPO.load(PPO_Path, env=env) #reload model after being deleted
evaluate_policy(model, env, n_eval_episodes=10, render=True) #evaluating 10 episodes of newly learned model
env.close()

training_log_path = os.path.join(log_path, 'PPO_2')
#print(training_log_path)

'''
#!tensorboard --logdir={training_log_path}
from tensorboard import program

tracking_address = training_log_path # the path of your log file.

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
'''

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

save_path = os.path.join('Training', 'Saved Models')
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=10000, best_model_save_path=save_path, verbose=1)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000, callback=eval_callback)

net_arch=[dict(pi=[128, 128, 128, 128], vf=[128, 128, 128, 128])] #dictionary for custom actor with 128 units in each layer, and same for value function vf
model = PPO('MlpPolicy', env, verbose = 1, policy_kwargs={'net_arch': net_arch})
model.learn(total_timesteps=20000, callback=eval_callback)

#CHECK CUSTOM ALGORITHM EVAL IN BASELINES PAGE

#using an alternate algorithm(dqn) instead of mlp policy

from stable_baselines3 import DQN

model = DQN('MlpPolicy', env, verbose = 1, tensorboard_log=log_path)
model.learn(total_timesteps=20000, callback=eval_callback)
dqn_path = os.path.join('Training', 'Saved Models', 'DQN_model')
model.save(dqn_path)
model = DQN.load(dqn_path, env=env)



