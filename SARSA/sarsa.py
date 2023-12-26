import pickle
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def run(episodes,is_training=True,render=True):
    env = gym.make('CliffWalking-v0',render_mode='human' if render else None)

    if is_training:
        q = np.zeros((env.observation_space.n,env.action_space.n))
    else:
        f = open('CliffWalking.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9
    discounted_factor_g = 0.9

    epsilon = 1
    epsilon_decay_rate = 1/episodes
    rng = np.random


    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        rewards = 0

        while not terminated and rewards>-100:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
                action_value = q[state,action]
                while action_value == -100:
                    action = env.action_space.sample()
                    action_value = q[state, action]
            else:
                action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training and rng.random() < epsilon:
                next_action = env.action_space.sample()
                next_action_value = q[new_state,next_action]
                while next_action_value == -100:
                    next_action = env.action_space.sample()
                    next_action_value = q[state, action]

            else:
                next_action = np.argmax(q[new_state,:])
            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (reward + discounted_factor_g * q[new_state, next_action]
                                                                        - q[state, action])
            state = new_state
            action = next_action
            rewards += reward
            epsilon = max(epsilon-epsilon_decay_rate,0)
            rewards_per_episode[i] = rewards

    env.close()

    if is_training:
        f = open('CliffWalking.pkl','wb')
        pickle.dump(q,f)
        f.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0,t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig(f'CliffWalking.png')

if __name__ == '__main__':
    #Goi ham traning
    #run(1000,is_training=True,render=True)
    #Thuc thi sau khi train
    run(10, is_training=False, render=True)
