import random, time


class SarsaAgent:

    def __init__(self, env, policy, alpha=0.2, gamma=0.9):
        assert alpha >= 0, "alpha non valido"
        assert 0 <= gamma < 1, "gamma non valido"
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.env = env
        self.actions = [i for i in range(0, self.env.get_action_space().n)]
        self.policy = policy

    def add_q_value(self, state, action, q_value):
        self.q_table[(tuple(state), action)] = q_value

    def get_q_value(self, state, action):
        return self.q_table.get((tuple(state), action), 0.0)

    def learn(self, state, action, reward, new_state, new_action, done):
        if (done is True):
            '''Terminal state, Q(s',a') not exists'''
            q_succ = 0
        else:
            ''' Q(s',a')'''
            q_succ = self.get_q_value(new_state, new_action)
        '''R + gamma * Q(s',a')'''
        learned_value = reward + self.gamma * q_succ
        '''Q(s,a)'''
        actual_q_value = self.get_q_value(state, action)
        '''Q(s,a) + alpha * [R + gamma * Q(s',a') - Q(s,a)]'''
        q_value = actual_q_value + self.alpha * learned_value - self.alpha * actual_q_value
        self.add_q_value(state, action, q_value)

    def choose_action(self, state):
        q_list = [self.get_q_value(state, action) for action in self.actions]
        action = self.policy.compute_action(q_list)
        return action

    def get_q_table(self):
        return self.q_table

    def set_policy(self, policy):
        self.policy = policy

    def get_policy(self):
        return self.policy

    def start_training(self, num_of_episodes=100, time_between_step=1, time_between_episode=1):
        assert num_of_episodes > 0, "number_of_episodes non valido"
        self.env.initialize_env()
        res = []
        for episode in range(0, num_of_episodes):
            rewards = []
            infos = []
            print("Started episode ", episode)
            time.sleep(time_between_episode)
            state, reward, done, info = self.env.reset()
            action = self.choose_action(state)
            rewards.append(reward)
            infos.append(info)
            while (done is False):
                new_state, reward, done, info = self.env.step(action)
                new_action = self.choose_action(new_state)
                self.learn(state, action, reward, new_state, new_action, done)
                action = new_action
                state = new_state
                rewards.append(reward)
                infos.append(info)
                time.sleep(time_between_step)
            res.append([rewards, infos])
        return res
