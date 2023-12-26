import pickle
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def run(episodes,is_training=True,render=True):
    env = gym.make('CliffWalking-v0',render_mode='human' if render else None)

    if is_training:
        q = np.zeros((env.observation_space.n,env.action_space.n))
    else:
        f = open('CliffWalkingNoPolicy.pkl','rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9
    discounted_factor_g = 0.9

    epsilon = 1
    epsilon_decay_rate = 1/episodes

    rng = np.random
    rewards_per_episodes = np.zeros(episodes)
    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        rewards = 0

        while not terminated and rewards>-100:
            if is_training and rng.random() <epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state,:])
            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (reward + discounted_factor_g * max(q[new_state,:])
                                                                   - q[state,action])
            state = new_state
            epsilon = max(epsilon-epsilon_decay_rate,0)
            rewards += reward
            rewards_per_episodes[i] = rewards

    env.close()

    if is_training:
        f = open('CliffWalkingNoPolicy.pkl','wb')
        pickle.dump(q,f)
        f.close()

    sum_reward = np.zeros(episodes)
    for t in range(episodes):
        sum_reward[t] = np.sum(rewards_per_episodes[max(0, t - 100):(t + 1)])
        plt.plot(sum_reward)
        plt.savefig(f'CliffWalkingNoPolicy.png')

if __name__ == '__main__':
    #run(1000,is_training=True,render=False)
    run (10,is_training=False,render=True)

