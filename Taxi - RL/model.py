import numpy as np
import pickle
import gym


class Agent:
    def __init__(self, env, table_path):
        self.env = env
        self.table_path = table_path
        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])  # 500x6

        self.alpha = .1
        self.gamma = .6
        self.epsilon = 1
        self.epsilon_min = .01
        self.epsilon_decay = .9995

        # information for each episode
        self.action = None
        self.state = None
        self.done = None
        self.epochs = 0
        self.rewards = 0
        self.penalties = 0  # pickup or dropoff the passenger at wrong location
        self.accidents = 0  # go to the wall

    def run_episodes(self, episodes, log_per_episode=10):
        for episode in range(1, episodes+1):
            self.state = self.env.reset()
            self.epochs, self.rewards, self.penalties, self.accidents = 0, 0, 0, 0
            self.done = False

            while not self.done:
                self.update_epsilon()
                self.action = self.get_action(self.epsilon)
                new_state, reward, done = self.do_action(self.action)
                self.update_table(new_state, reward)

                self.state = new_state
                self.rewards += reward
                self.done = done
                self.epochs += 1
                if reward == -10:
                    self.penalties += 1
                elif reward == -3:
                    self.accidents += 1

            if not episode % log_per_episode:  # every <log_per_episode>
                print('Episode: ', episode)
                print('Epochs: ', self.epochs)
                print('Rewards: ', self.rewards)
                print('Penalties: ', self.penalties)
                print('Accidents: ', self.accidents)
                print('\n', '-'*50)

    def update_table(self, new_state, reward):
        old_value = self.q_table[self.state, self.action]
        next_max = np.max(self.q_table[new_state])
        new_value = ((1 - self.alpha) * old_value) + (self.alpha * (reward + (self.gamma * next_max)))
        self.q_table[self.state, self.action] = new_value

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def get_action(self, epsilon=.5):
        if random_run(epsilon):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[self.state])

    def do_action(self, action):
        new_state, reward, done, *info = self.env.step(action)
        if (new_state == self.state) and (reward == -1):  # punish for accident with wall
            reward = -3

        return new_state, reward, done

    def save_table(self, path=None):
        if not path:
            path = self.table_path

        with open(path, 'wb') as handler:
            return pickle.dump(self.q_table, handler)

    def load_table(self, path=None):
        if not path:
            path = self.table_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"File <{path}> doesn't exists!")

        with open(path, 'rb') as handler:
            self.q_table = pickle.load(handler)

        print(f'Table successfully loaded from <{path}>')


def random_run(probability=.5):
    if np.random.uniform(0, 1) < probability:
        return True
    return False


if __name__ == '__main__':
    env = gym.make('Taxi-v3', render_mode='ansi').env
    agent = Agent(env, table_path='QTable.h5')

    try:
        agent.load_table()
    except Exception:
        pass

    agent.run_episodes(episodes=1000, log_per_episode=50)
    agent.save_table()
