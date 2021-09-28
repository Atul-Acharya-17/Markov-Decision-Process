import numpy as np
import random

class ExpectedSarsa():

    def __init__(self, n_w, n_h, n_actions, gamma=0.99, step_size=0.1, num_episodes=50000, num_steps=100, epsilon=0.1) -> None:
        self.q_table = np.zeros((n_h, n_w, n_actions))
        self.n_h = n_h
        self.n_w = n_w
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = step_size
        self.num_episodes=num_episodes
        self.num_steps=num_steps
        self.epsilon = epsilon

    def solve(self, mdp):

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

                prob = np.full((4,), self.epsilon / self.n_actions)
                prob[np.argmax(self.q_table[next_state[0]][next_state[1]])] = 1 - self.epsilon + self.epsilon / self.n_actions

                v_value = np.dot(self.q_table[next_state[0]][next_state[1]], prob)

                self.q_table[state[0]][state[1]][action] = self.q_table[state[0]][state[1]][action] + self.alpha * (reward + self.gamma * v_value - self.q_table[state[0]][state[1]][action])

                state = next_state

        return self.q_table

    def get_starting_state(self, mdp, n_h, n_w):

        state = (random.randint(0, n_h-1), random.randint(0, n_w-1))

        while mdp.is_wall(state):
            state = (random.randint(0, n_h-1), random.randint(0, n_w-1))
        
        return state