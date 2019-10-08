import gym
import gym_maze
from GreedyPolicies import *
from BoltzmannPolicies import *
from QLearningAgent import QLearningAgent
from SarsaAgent import SarsaAgent

env = gym.make('Maze-v0')
greed = GreedyPolicy()
e_greedy = EGreedyPolicy(decay=True, epsilon=0.3)
boltzmann = BoltzmannPolicy()
boltzmann_e_greedy = BoltzmannEGreedyPolicy()

def train_ql_agent():
    qLearningAgent = QLearningAgent(env=env, policy=e_greedy, number_of_episodes=5)
    qLearningAgent.start_training(time_between_step=0.1, time_between_episode=1)

def train_sarsa_agent():
    sarsaAgent = SarsaAgent(env=env, policy=e_greedy, number_of_episodes=5)
    sarsaAgent.start_training(time_between_step=0.1, time_between_episode=1)

def main():
    train_ql_agent()
    train_sarsa_agent()

main()
