from model import Agent

import gym


env = gym.make('Taxi-v3', render_mode='human', new_step_api=True).env  # graphical render mode by pygame
agent = Agent(env, table_path='QTable.h5')
agent.load_table()

while True:
    episodes = int(input('Several episodes: '))
    agent.run_episodes(episodes=episodes, log_per_episode=1)

    if input('Do you want try again? (y/n) ').lower() == 'n':
        break
