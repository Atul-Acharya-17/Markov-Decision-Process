import numpy as np
import random


class MonteCarlo():

    def __init__(self, n_w, n_h, n_actions, gamma=0.99, num_episodes=100000, num_steps=1000) -> None:
        self.q_table = np.zeros((n_h, n_w, n_actions))
        self.n_h = n_h
        self.n_w = n_w
        self.n_actions = n_actions
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.epsilon = 0.1

    def solve(self, mdp):

        # Solve and return Q_table
        returns_count = {}

        for episode in range(self.num_episodes):

            states = []
            actions = []
            rewards = []

            state = self.get_starting_state(mdp, self.q_table.shape[0], self.q_table.shape[1])

            G = 0

            seen = {}

            for step in range(self.num_steps):

                rand_num = random.uniform(0, 1)
                if rand_num <= self.epsilon:
                    action = random.randint(0, self.n_actions-1)
                else:
                    action = np.argmax(self.q_table[state[0]][state[1]])

                next_state, reward = mdp.step(state, mdp.action_map(action))

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                if (state, action) in seen:
                    seen[state, action] += 1
                else:
                    seen[state, action] = 1

                state = next_state

            # Loop Backwards
            for i in range(len(states)-1, -1, -1):
                state = states[i]
                action = actions[i]
                reward = rewards[i]
                G = self.gamma * G + reward
                
                # First Visit
                if seen[(state, action)] > 1:
                    seen[(state, action)] -= 1
                    continue
                
                if (state, action) in returns_count:
                    self.q_table[state[0]][state[1]][action] = (self.q_table[state[0]][state[1]][action] * returns_count[(state, action)] + G) / (1 + returns_count[(state, action)])
                    returns_count[(state, action)] += 1

                else:
                    self.q_table[state[0]][state[1]][action] = G
                    returns_count[(state, action)] = 1

        return self.q_table

    def get_starting_state(self, mdp, n_h, n_w):

        state = (random.randint(0, n_h-1), random.randint(0, n_w-1))

        while mdp.is_wall(state):
            state = (random.randint(0, n_h-1), random.randint(0, n_w-1))
        
        return state