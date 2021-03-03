from ..environment.env import Environment
import numpy as np


class ValueIteration(Environment):

    def __init__(self, grid_world, actions, rewards, gw, gh, gamma):
        super(ValueIteration, self).__init__(grid_world, actions, rewards, gw, gh)
        self.gamma = gamma
        self.utilities = np.zeros((gh, gw))

        self.data = {}

        for i in range(self.utilities.shape[0]):
            for j in range(self.utilities.shape[1]):
                cur_state = (i, j)
                self.data[f'{cur_state}'] = [0] 


    def solve(self, num_iterations, threshold):
        for iteration in range(num_iterations):
            new_utilities = self.utilities.copy()
            for i in range(self.utilities.shape[0]):
                for j in range(self.utilities.shape[1]):
                    cur_state = (i, j)
                    if self.is_wall(cur_state):
                        self.data[f'{cur_state}'].append(0)
                        continue

                    actions = self.get_actions()
                    
                    action_values = []
                    for action in actions:

                        action_value = 0
                        transition_model = self.transition_model(cur_state, action)
                        for next_state in transition_model:
                            utility = self.utilities[next_state[0]][next_state[1]]
                            probability = transition_model[next_state]
                            expected_utility = probability * utility
                            action_value += expected_utility
                        
                        action_values.append(action_value)

                    reward = self.receive_reward(cur_state)
                    state_value = reward + self.gamma * max(action_values)
                    new_utilities[i][j] = state_value
                    self.data[f'{cur_state}'].append(state_value)
            self.utilities = new_utilities.copy()

        return self.utilities

    def get_data(self):
        return self.data
