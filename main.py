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
    q_table_file_name = "q_learning_table.pkl"
    qLearningAgent = QLearningAgent(env=env, policy=e_greedy)
    qLearningAgent.load_stored_q_table(q_table_file_name)
    qLearningAgent.start_training(num_of_episodes=5, time_between_step=0.1, time_between_episode=1, save_q_table=True, q_table_file_name=q_table_file_name)

def train_sarsa_agent():
    sarsa_file_name = "sarsa_table.pkl"
    sarsaAgent = SarsaAgent(env=env, policy=e_greedy)
    sarsaAgent.load_stored_q_table(sarsa_file_name)
    sarsaAgent.start_training(num_of_episodes=5, time_between_step=0.1, time_between_episode=1, save_q_table=True, q_table_file_name=sarsa_file_name)

def main():
    train_ql_agent()
    train_sarsa_agent()

main()
