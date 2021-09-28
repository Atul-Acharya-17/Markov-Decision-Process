import numpy as np
import random
from copy import deepcopy


class DynaQ(object):

    def __init__(self, n_w, n_h, n_actions, gamma=0.99, step_size=0.1, num_episodes=50000, num_steps=100, planning_steps=100, epsilon=0.1) -> None:
        super().__init__()
        self.q_table = np.zeros((n_h, n_w, n_actions))
        self.n_h = n_h
        self.n_w = n_w
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = step_size
        self.num_episodes=num_episodes
        self.num_steps=num_steps
        self.epsilon = epsilon
        self.planning_steps=planning_steps
    
    def solve(self, mdp):

        model = {}

        # Solve and return Q_table
        for episode in range(self.num_episodes):

            state = self.get_starting_state(mdp, self.q_table.shape[0], self.q_table.shape[1])

            for step in range(self.num_steps):
                
                rand_num = random.uniform(0, 1)
                if rand_num <= self.epsilon:
                    action = random.randint(0, self.n_actions-1)
                else:
                    action = np.argmax(self.q_table[state[0]][state[1]])

                next_state, reward = mdp.step(state, mdp.action_map(action))

                self.q_table[state[0]][state[1]][action] = self.q_table[state[0]][state[1]][action] + self.alpha * (reward + self.gamma * np.max(self.q_table[next_state[0]][next_state[1]]) - self.q_table[state[0]][state[1]][action])

                if (state, action) not in model:
                    model[(state, action)] = {(next_state, reward): 1}

                elif (next_state, reward) not in model[(state, action)]:
                    model[(state, action)] = {(next_state, reward): 1}

                else:
                    model[(state, action)][(next_state, reward)] += 1

                state = next_state

            self.plan(model)
            

        return self.q_table

    def plan(self, model):

        for step in range(self.planning_steps):

            state_action_pair = random.choice(list(model.keys()))

            state = state_action_pair[0]
            action = state_action_pair[1]

            transition = model[state_action_pair]

            most_frequent = None
            for key, frequency in transition.items():
                if most_frequent is None:
                    most_frequent = key
                elif transition[most_frequent] < frequency:
                    most_frequent = key

            next_state = most_frequent[0]
            reward = most_frequent[1]
            
            self.q_table[state[0]][state[1]][action] = self.q_table[state[0]][state[1]][action] + self.alpha * (reward + self.gamma * np.max(self.q_table[next_state[0]][next_state[1]]) - self.q_table[state[0]][state[1]][action])

    def get_starting_state(self, mdp, n_h, n_w):
        state = (random.randint(0, n_h-1), random.randint(0, n_w-1))

        while mdp.is_wall(state):
            state = (random.randint(0, n_h-1), random.randint(0, n_w-1))
        
        return state
