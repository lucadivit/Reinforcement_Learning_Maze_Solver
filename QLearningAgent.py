import random


class QLearningAgent:

    def __init__(self, actions, policy, alpha=0.2, gamma=0.9):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions
        self.policy = policy

    def add_q_value(self, state, action, q_value):
        self.q_table[(tuple(state), action)] = q_value

    def get_q_value(self, state, action):
        return self.q_table.get((tuple(state), action), 0.0)

    def learn(self, state, action, reward, new_state, done):
        if(done is True):
            '''Terminal state, max(Q(s',a')) not exists'''
            max_q = 0
        else:
            ''' max(Q(s',a'))'''
            max_q = max(self.get_q_value(new_state, new_action) for new_action in self.actions)
        '''R + gamma * max(Q(s',a'))'''
        learned_value = reward + self.gamma * max_q
        '''Q(s,a)'''
        actual_q_value = self.get_q_value(state, action)
        '''Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]'''
        q_value = actual_q_value + self.alpha * learned_value - self.alpha * actual_q_value
        self.add_q_value(state, action, q_value)

    def choose_action(self, state):
        q_list = [self.get_q_value(state, action) for action in self.actions]
        action = self.policy.compute_action(q_list)
        return action

    def get_q_table(self):
        return self.q_table

