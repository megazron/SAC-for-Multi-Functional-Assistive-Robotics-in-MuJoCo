import os
import glob
import time
from datetime import datetime
from tqdm import tqdm  # For progress bar

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import gymnasium as gym

from SAC import SAC, device

# Generate GIF and All Visuals 
def generate_gif_and_visuals(env_name, checkpoint_path, timestep, max_ep_len, is_final=False, log_f_name=None):
    gif_dir = "SAC_gifs/" + env_name + '/'
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

    gif_images_dir = "SAC_gif_images/" + env_name + '/checkpoint_' + str(timestep) + '/'
    if not os.path.exists(gif_images_dir):
        os.makedirs(gif_images_dir)

    fig_dir = "SAC_figs/" + env_name + '/checkpoint_' + str(timestep) + '/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    env = gym.make(env_name, render_mode='rgb_array')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    sac_agent = SAC(state_dim, action_dim, 0.0003, 0.0003, 0.99, 0.005, 0.2, 1, True, 256, int(1e6), 0, env.action_space)
    sac_agent.load(checkpoint_path)

    state, _ = env.reset()
    images = []

    for t in range(1, max_ep_len+1):
        action = sac_agent.select_action(state, evaluate=True)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        img = env.render()
        images.append(Image.fromarray(img))

        if done:
            break

    env.close()

    gif_name = 'final.gif' if is_final else f'checkpoint_{timestep}.gif'
    gif_path = gif_dir + gif_name
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=50, loop=0, optimize=True)
    print(f"GIF saved at {gif_path}")

    if log_f_name:
        df = pd.read_csv(log_f_name)

        # Reward Curve
        plt.figure(figsize=(10, 6))
        plt.plot(df['timestep'], df['reward'], label='Average Reward', color='blue')
        plt.title(f"{env_name} Reward Curve (Checkpoint {timestep})")
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(fig_dir + "reward_curve.png")
        plt.close()

        # Smoothed Reward
        df['smoothed'] = df['reward'].rolling(window=50).mean()
        plt.figure(figsize=(10, 6))
        plt.plot(df['timestep'], df['smoothed'], label='Smoothed Reward', color='green')
        plt.title(f"{env_name} Smoothed Reward (Checkpoint {timestep})")
        plt.xlabel("Timestep")
        plt.ylabel("Smoothed Reward")
        plt.legend()
        plt.savefig(fig_dir + "smoothed_reward.png")
        plt.close()

        # Reward Histogram
        plt.figure(figsize=(8, 6))
        plt.hist(df['reward'], bins=20, color='purple', alpha=0.7)
        plt.title(f"{env_name} Reward Distribution (Checkpoint {timestep})")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.savefig(fig_dir + "reward_histogram.png")
        plt.close()

        # Reward Heatmap (dynamic)
        data_len = len(df['reward'])
        cols = min(10, data_len)
        if cols > 0:
            rows = max(1, data_len // cols)
            reward_matrix = np.array(df['reward'][:rows * cols]).reshape(rows, cols)
            plt.figure(figsize=(10, 6))
            sns.heatmap(reward_matrix, cmap='viridis', annot=False)
            plt.title(f"{env_name} Reward Heatmap (Checkpoint {timestep})")
            plt.xlabel("Group")
            plt.ylabel("Episode")
            plt.savefig(fig_dir + "reward_heatmap.png")
            plt.close()

        print("All visuals generated in " + fig_dir)

# Training
def train():
    print("============================================================================================")

    # initialize environment hyperparameters 
    env_name = "InvertedPendulum-v4"  # change the name to your desired model

    has_continuous_action_space = True

    max_ep_len = 1000
    max_training_timesteps = int(2.5e6)  # change this based on the model

    print_freq = 5000  
    log_freq = 500     
    save_model_freq = int(1e4)  # change for More checkpoints for GIFs/visuals

  

    #SAC hyperparameters 
    lr_actor = 0.0003
    lr_critic = 0.0003
    gamma = 0.99
    tau = 0.005
    alpha = 0.2
    target_update_interval = 1
    automatic_entropy_tuning = True
    hidden_size = 256  # change this based on the model
    replay_size = int(1e6)
    batch_size = 256 
    updates_per_step = 1
    start_steps = 5000
    random_seed = 0
   
    print("training environment name : " + env_name)

    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # logging 
    log_dir = "SAC_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    run_num = len(next(os.walk(log_dir))[2])
    log_f_name = log_dir + '/SAC_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    

    # checkpointing 
    run_num_pretrained = 0

    directory = "SAC_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)

    checkpoint_path = directory + "SAC_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    
    # print all hyperparameters 
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    print("Initializing a continuous action space policy (SAC)")
    print("--------------------------------------------------------------------------------------------")
    print("SAC gamma : ", gamma)
    print("SAC tau : ", tau)
    print("SAC alpha : ", alpha)
    print("SAC target_update_interval : ", target_update_interval)
    print("SAC automatic_entropy_tuning : ", automatic_entropy_tuning)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
   

    print("============================================================================================")

    sac_agent = SAC(state_dim, action_dim, lr_actor, lr_critic, gamma, tau, alpha, target_update_interval, automatic_entropy_tuning, hidden_size, replay_size, random_seed, env.action_space)

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    episode_rewards = []

    # Progress bar with ETA
    pbar = tqdm(total=max_training_timesteps, desc="Training Progress", unit="step", position=0, leave=True)

    while time_step <= max_training_timesteps:
        state, _ = env.reset()
        current_ep_reward = 0
        t = 0

        while True:
            t += 1
            if time_step < start_steps:
                action = env.action_space.sample()
            else:
                action = sac_agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            sac_agent.memory.push(state, action, reward, next_state, done)

            state = next_state
            current_ep_reward += reward
            time_step += 1
            pbar.update(1)  

            if len(sac_agent.memory) > batch_size:
                for _ in range(updates_per_step):
                    sac_agent.update(batch_size)

            if time_step % log_freq == 0:
                if log_running_episodes > 0:
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)
                    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    log_f.flush()
                log_running_reward = 0
                log_running_episodes = 0

            if time_step % print_freq == 0:
                if print_running_episodes > 0:
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)
                    print("\nEpisode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
                print_running_reward = 0
                print_running_episodes = 0

            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                sac_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("Generating GIF and visuals for checkpoint at timestep", time_step)
                generate_gif_and_visuals(env_name, checkpoint_path, time_step, max_ep_len, is_final=False, log_f_name=log_f_name)
                print("GIF and visuals generated")
                print("--------------------------------------------------------------------------------------------")

            if done or t >= max_ep_len:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        episode_rewards.append(current_ep_reward)

        i_episode += 1

    pbar.close()

    log_f.close()
    env.close()

    # Final GIF and visuals
    print("Generating final GIF and visuals")
    generate_gif_and_visuals(env_name, checkpoint_path, max_training_timesteps, max_ep_len, is_final=True, log_f_name=log_f_name)
    print("Final GIF and visuals generated")

    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

if __name__ == '__main__':
    train()
