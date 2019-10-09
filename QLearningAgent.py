import random, time


class QLearningAgent:

    def __init__(self, env,  policy, number_of_episodes=100, alpha=0.2, gamma=0.9):
        assert number_of_episodes > 0, "number_of_episodes non valido"
        assert alpha >= 0, "alpha non valido"
        assert 0 <= gamma < 1, "gamma non valido"
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.env = env
        self.actions = [i for i in range(0, self.env.get_action_space().n)]
        self.policy = policy
        self.nb_episodes = number_of_episodes

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

    def start_training(self, time_between_step=1, time_between_episode=1):
        self.env.initialize_env()
        res = []
        for i in range(0, self.nb_episodes):
            rewards = []
            infos = []
            print("Started episode ", i)
            time.sleep(time_between_episode)
            state, reward, done, info = self.env.reset()
            rewards.append(reward)
            infos.append(info)
            while (done is False):
                action = self.choose_action(state)
                new_state, reward, done, info = self.env.step(action)
                self.learn(state, action, reward, new_state, done)
                state = new_state
                rewards.append(reward)
                infos.append(info)
                time.sleep(time_between_step)
            res.append([rewards, infos])
        return res

    def get_q_table(self):
        return self.q_table

    def set_policy(self, policy):
        self.policy = policy

    def get_policy(self):
        return self.policy
