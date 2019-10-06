import random


class EGreedyPolicy:

    def __init__(self, epsilon=0.2, decay=False):
        assert epsilon >= 0, "Valore epsilon non valido."
        assert decay is True or decay is False, "Valore per decay non valido."
        self.epsilon = epsilon
        self.decay = decay
        self.actions = None
        self.decay_factor = self.epsilon/1000

    def compute_action(self, q_list):
        if(self.actions is None):
            self.actions = [i for i in range(len(q_list))]
        action = None
        if(random.random() < self.epsilon):
            action = random.choice(self.actions)
        else:
            max_q = max(q_list)
            num_of_same_elments = q_list.count(max_q)
            if(num_of_same_elments > 1):
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