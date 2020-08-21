import gym
import collections
import numpy as np

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.nA = self.env.action_space.n
        self.qvalues = collections.defaultdict(lambda: np.zeros(self.nA))
        
    def greedy_policy(self, epsilon=0.1):
        def policy_fn(observation):
            A = np.ones(self.nA, dtype=float) * epsilon / self.nA 
            best_action = np.argmax(self.qvalues[observation])
            A[best_action] += (1.0 - epsilon)
            return A
        return policy_fn
    
    def choose_action(self, state):
        action_probs = self.greedy_policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def _update(self, state, action, reward, next_state):
        """Update according to Q-learning algorithm"""
        best_next_action = np.argmax(self.qvalues[next_state])    
        td_target = reward + GAMMA * self.qvalues[next_state][best_next_action]
        td_delta = td_target - self.qvalues[state][action]
        self.qvalues[state][action] += ALPHA * td_delta
    
    def train(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.choose_action(state)
            next_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            
            self._update(state, action, reward, next_state)
            if is_done:
                break
            state = next_state
        return total_reward
    
    def play_episode(self, env):
        """Play episode without update q values"""
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.choose_action(state)
            next_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = next_state
        return total_reward


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()

    num_episode = 0
    best_reward = 0.0
    while True:
        num_episode += 1
        agent.train(test_env)
        reward = 0.0
        
        # Test each 10 episodes
        if num_episode % 10 == 0:
            for _ in range(TEST_EPISODES):
                reward += agent.play_episode(test_env)
            reward /= TEST_EPISODES
            if reward > best_reward:
                print("Best reward updated %.3f -> %.3f" % (
                    best_reward, reward))
                best_reward = reward
                
        if reward > 0.80:
            print("Solved in %d episodes!" % num_episode)
            break