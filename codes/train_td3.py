import os
import gym
import torch
from src.td3.td3 import TD3
from src.td3.replayBuffer import ReplayBuffer
from src.td3.evaluate import evaluate_policy
from src.td3.utils import (create_folders, mkdir)
from src.td3.visualizations import training_td3_results
from src.config import arguments_parser

args = arguments_parser()

# parametros
env_name = args.env_name
seed = args.seed
start_timesteps = args.start_timesteps
eval_freq = args.eval_freq
max_timesteps = args.max_timesteps
save_models = args.save_models
expl_noise = args.expl_noise
batch_size = args.batch_size
discount = args.discount
tau = args.tau
policy_noise = args.policy_noise
noise_clip = args.noise_clip
policy_freq = args.policy_freq


# configuración del entorno
file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
print("---------------------------------------")
print("Configuración: %s" % (file_name))
print("---------------------------------------")
env = gym.make(env_name)
