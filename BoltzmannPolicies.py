import math, random

class BoltzmannPolicy:

    def __init__(self, actions=None, tau=500, tau_decay=False, decay_factor_tau=0.1):
        assert tau >= 0, "Valore tau non valido."
        assert bool == type(tau_decay), "Valore per tau decay non valido."
        self.actions = actions
        self.tau = tau
        ''' tau --> 0 pure exploitation
            tau --> infty pure exploration
        '''
        self.tau_decay = tau_decay
        self.decay_factor_tau = decay_factor_tau

    def compute_action(self, q_list):
        if(self.actions is None):
            self.actions = [i for i in range(len(q_list))]
        assert len(self.actions) == len(q_list), "Q values list is not len as actions list."
        '''P(a)=exp(Q(s,a)/tau)/sum_t(exp(Q(s,a_t)/tau))'''
        probs = []
        for q in q_list:
            '''num = exp(Q(s,a)/tau)'''
            num = math.exp(q/self.tau)
            exp_values_for_dem = []
            for q_t in q_list:
                '''exp(Q(s,a_t)/tau)'''
                exp_values_for_dem.append(math.exp(q_t/self.tau))
            '''den = sum_t(exp(Q(s,a_t)/tau))'''
            den = sum(exp_values_for_dem)
            '''num/den'''
            probs.append(round(num/den, 10))
        if (self.tau_decay is True):
            new_tau = self.tau - self.decay_factor_tau
            if (new_tau < 0):
                self.tau = 0
            else:
                self.tau = new_tau
        return random.choices(self.actions, weights=probs, k=1)[0]


class BoltzmannEGreedyPolicy:
    def __init__(self, actions=None, epsilon=0.2, eps_decay=False, tau=500, tau_decay=False, decay_factor_epsilon=0.001, decay_factor_tau=0.1):
        assert tau >= 0, "Valore tau non valido."
        assert 0 <= epsilon <= 1, "Valore epsilon non valido."
        assert bool == type(eps_decay), "Valore per decay non valido."
        assert bool == type(tau_decay), "Valore per tau decay non valido."
        assert 0 <= decay_factor_epsilon <= 1, "Valore per decay factor epsilon non valido."
        self.eps_decay = eps_decay
        self.epsilon = epsilon
        self.actions = actions
        self.tau = tau
        self.decay_factor_epsilon = decay_factor_epsilon
        self.decay_factor_tau = decay_factor_tau
        self.tau_decay = tau_decay

    def compute_action(self, q_list):
        if(self.actions is None):
            self.actions = [i for i in range(len(q_list))]
        assert len(self.actions) == len(q_list), "Q values list is not len as actions list"
        if (random.random() <= self.epsilon):
            max_q = max(q_list)
            num_of_same_elements = q_list.count(max_q)
            if (num_of_same_elements > 1):
                idx_of_same_elements = [i for i in range(0, len(self.actions)) if max_q == q_list[i]]
                idx = random.choice(idx_of_same_elements)
                action = self.actions[idx]
            else:
                action = self.actions[q_list.index(max_q)]
        else:
            probs = []
            for q in q_list:
                '''num = exp(Q(s,a)/tau)'''
                num = math.exp(q / self.tau)
                exp_values_for_dem = []
                for q_t in q_list:
                    '''exp(Q(s,a_t)/tau)'''
                    exp_values_for_dem.append(math.exp(q_t / self.tau))
                '''den = sum_t(exp(Q(s,a_t)/tau))'''
                den = sum(exp_values_for_dem)
                '''num/den'''
                probs.append(round(num / den, 10))
            action = random.choices(self.actions, weights=probs, k=1)[0]
            if(self.tau_decay is True):
                new_tau = self.tau - self.decay_factor_tau
                if(new_tau < 0):
                    self.tau = 0
                else:
                    self.tau = new_tau
        if(self.eps_decay is True):
            new_epsilon = self.epsilon - self.decay_factor_epsilon
            if(new_epsilon < 0):
                self.epsilon = 0
            else:
                self.epsilon = new_epsilon
        return action
