import gym
import gym_maze
from EGreedyPolicy import EGreedyPolicy
from QLearningAgent import QLearningAgent

env = gym.make('Maze-v0')
policy = EGreedyPolicy(decay=True, epsilon=0.3)


def train_ql_agent():
    qLearningAgent = QLearningAgent(env, policy, 5)
    for _ in range(2):
        qLearningAgent.start_training(time_between_step=1, time_between_episode=1)


def main():
    train_ql_agent()

main()
