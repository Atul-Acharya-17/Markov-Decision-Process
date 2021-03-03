import numpy as np
import copy
import math
from ..environment.env import Environment


#TODO: change the data dictionary to column, row instead

class PolicyIteration(Environment):

    def __init__(self, grid_world, actions, rewards, gw, gh, gamma):
        super(PolicyIteration, self).__init__(grid_world, actions, rewards, gw, gh)
        self.gamma = gamma
        self.utilities = np.zeros((gh, gw))
        self.policy = self.get_starting_policy()
        self.utilities = np.zeros((gh, gw))

        self.data = {}

        for i in range(self.utilities.shape[0]):
            for j in range(self.utilities.shape[1]):
                cur_state = (i, j)
                self.data[f'{cur_state}'] = [0] 


    def solve(self, num_iterations, threshold):
        stable_policy = False
        while not stable_policy:
            self.policy_evaluation(num_iterations)
            stable_policy = self.policy_improvement()
        return self.utilities
    
    def policy_evaluation(self, num_iterations):
        for iteration in range(num_iterations):
            new_utilities = self.utilities
            for i in range(self.utilities.shape[0]):
                for j in range(self.utilities.shape[1]):
                    cur_state = (i, j)
                    if self.is_wall(cur_state):
                        self.data[f'{cur_state}'].append(0)
                        continue
                    action = self.policy[i][j]
                    transition_model = self.transition_model(cur_state, action)
                    action_value = 0
                    for next_state in transition_model:
                        utility = self.utilities[next_state[0]][next_state[1]]
                        probability = transition_model[next_state]
                        expected_value = probability * utility
                        action_value += expected_value
                    reward = self.receive_reward(cur_state)
                    utility = reward + self.gamma * action_value
                    new_utilities[i][j] = utility
                    self.data[f'{cur_state}'].append(utility)
            self.utilities = new_utilities

    def policy_improvement(self):
        new_policy = copy.deepcopy(self.policy)
        for i in range(self.utilities.shape[0]):
            for j in range(self.utilities.shape[1]):
                cur_state = (i, j)
                if self.is_wall(cur_state):
                        continue

                actions = self.actions
                action_values = {}
                for action in actions:
                    action_value = 0
                    transition_model = self.transition_model(cur_state, action)
                    for next_state in transition_model:
                        utility = self.utilities[next_state[0]][next_state[1]]
                        probability = transition_model[next_state]
                        expected_utility = probability * utility
                        action_value += expected_utility
                    action_values[action] = action_value
                best_action = None
                best_action_value = -math.inf
                for action in action_values:
                    if action_values[action] > best_action_value:
                        best_action = action
                        best_action_value = action_values[action]
                new_policy[i][j] = best_action

        for i in range(len(self.policy)):
            for j in range(len(self.policy[i])):
                if self.policy[i][j] != new_policy[i][j]:
                    self.policy[i][j] = new_policy[i][j]
                    return False
        
        self.policy[i][j] = new_policy[i][j]
        return True

    def get_starting_policy(self):
        policy = [[(1, 0) for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        return policy

    def get_data(self):
        return self.data

