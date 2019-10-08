import random

class GreedyPolicy:

    def __init__(self, actions=None):
        self.actions = actions

    def compute_action(self, q_list):
        if(self.actions is None):
            self.actions = [i for i in range(len(q_list))]
        assert len(self.actions) == len(q_list), "Q values list is not len as actions list"
        max_q = max(q_list)
        num_of_same_elements = q_list.count(max_q)
        if (num_of_same_elements > 1):
            idx_of_same_elements = [i for i in range(0, len(self.actions)) if max_q == q_list[i]]
            idx = random.choice(idx_of_same_elements)
            action = self.actions[idx]
        else:
            action = self.actions[q_list.index(max_q)]
        return action

class EGreedyPolicy:

    def __init__(self, actions=None, epsilon=0.2, decay=False, decay_factor=0.001):
        assert 0 <= epsilon <= 1, "Valore epsilon non valido."
        assert bool == type(decay), "Valore per decay non valido."
        assert 0 <= decay_factor <= 1, "Valore per decay factor non valido."
        self.epsilon = epsilon
        self.decay = decay
        self.actions = actions
        self.decay_factor = decay_factor

    def compute_action(self, q_list):
        if(self.actions is None):
            self.actions = [i for i in range(len(q_list))]
        assert len(self.actions) == len(q_list), "Q values list is not len as actions list"
        if(random.random() <= self.epsilon):
            action = random.choice(self.actions)
        else:
            max_q = max(q_list)
            num_of_same_elements = q_list.count(max_q)
            if(num_of_same_elements > 1):
                idx_of_same_elements = [i for i in range(0, len(self.actions)) if max_q == q_list[i]]
                idx = random.choice(idx_of_same_elements)
                action = self.actions[idx]
            else:
                action = self.actions[q_list.index(max_q)]
        if(self.decay is True):
            new_epsilon = self.epsilon - self.decay_factor
            if(new_epsilon < 0):
                self.epsilon = 0
            else:
                self.epsilon = new_epsilon
        return action