import gymnasium as gym

import keras
import numpy as np

env = gym.make('CartPole-v1',render_mode='human')
state, _ = env.reset()
state_size = env.observation_space.shape[0]

my_agent = keras.models.load_model('train_agent.h5')
n_timesteps = 500
total_reward = 0

for t in range(n_timesteps):
    env.render()

    state = state.reshape((1,state_size))
    q_values = my_agent.predict(state,verbose=0)
    max_q_values = np.argmax(q_values)

    next_state, reward, terminal, _, _ = env.step(action=max_q_values)
    total_reward += reward
    state = next_state
    print(t)