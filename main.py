import gym
import gym_maze
import time
from EGreedyPolicy import EGreedyPolicy
from QLearningAgent import QLearningAgent
from datetime import datetime
from matplotlib import pyplot as plt

results_file_name = "results.txt"
q_info_file_name = "q_table.txt"
file = None
q_info = None

env = gym.make('Maze-v0')
actions = [i for i in range(0, env.get_action_space().n)]
policy = EGreedyPolicy(actions, decay=True, epsilon=0.3)

def print_res_info(num_maze, start_time, end_time, num_mosse, num_partita, avg_reward):
    file.write("Maze Numero: " + str(num_maze) + "\n")
    file.write("\n")
    delta_time = datetime.strptime(end_time, "%H:%M:%S") - datetime.strptime(start_time, "%H:%M:%S")
    file.write("Termine Partita: " + str(num_partita) + "\n")
    file.write("Numero Mosse: " + str(num_mosse) + "\n")
    file.write("Reward Medio:" + str(avg_reward) + "\n")
    file.write("Tempo Impiegato: " + str(delta_time) + "\n")
    file.write("\n")
    file.flush()

def plot_infos(rewards, steps, num_maze):
    fig_name = "plot_maze_" + str(num_maze) + ".png"
    plt.ion()
    plt.title("Plot " + str(num_maze))
    fig = plt.figure(num_maze)
    plt.plot(rewards, color='green', label='rewards')
    plt.plot(steps, color='red', label='steps')
    plt.legend(loc='upper left')
    fig.savefig(fig_name)
    plt.close()

def print_q_info_info(num_maze, num_partita, info):
    q_info.write("Maze Numero: " + str(num_maze) + "\n")
    q_info.write("\n")
    q_info.write("Termine Partita: " + str(num_partita) + "\n")
    q_info.write(str(info) + "\n")
    q_info.write("\n")
    q_info.flush()

def train_qLearning_agent(num_mazes, num_episodes, time_from_ep, time_from_maze):
    global file
    global q_info
    qLearningAgent = QLearningAgent(actions, policy)
    file = open("q_learning_agent_" + results_file_name, "w")
    q_info = open("q_learning_agent_" + q_info_file_name, "w")
    env.initialize_env()
    for j in range(0, num_mazes):
        env.hard_reset()
        rewards_avg = []
        steps_avg = []
        for i in range(0, num_episodes):
            rewards = []
            steps = []
            state, reward, done, info = env.reset()
            start = datetime.now()
            start = start.strftime("%H:%M:%S")
            while (done is False):
                action = qLearningAgent.choose_action(state)
                new_state, reward, done, info = env.step(action)
                qLearningAgent.learn(state, action, reward, new_state, done)
                state = new_state
                rewards.append(reward)
                steps.append(info.get("num_steps"))
                time.sleep(time_from_ep)
            end = datetime.now()
            end = end.strftime("%H:%M:%S")
            avg_reward = sum(rewards)/float(len(rewards))
            print_res_info(j, start, end, info.get("num_steps"), i, avg_reward)
            print_q_info_info(j, i, qLearningAgent.get_q_table())
            print ("Termine Partita", i)
            rewards_avg.append(avg_reward)
            steps_avg.append(sum(steps)/float(len(steps)))
            time.sleep(time_from_maze)
        plot_infos(rewards_avg, steps_avg, j)

    file.close()
    q_info.close()

def main():
    train_qLearning_agent(2, 100, 0.05, 1)
    pass

main()
