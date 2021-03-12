from ..environment.env import Environment
import numpy as np
import math


'''
Value Iteration Class
This class inherits from the Environment class defined in environment.env.py
The parameters for the constructor are:

    * grid world - 2-D list that contains details of walls
    * actions - List of all possible actions
    * rewards - 2-D list representing the reward for each state
    * gw - Grid Width
    * gh - Grid Height
    * gamma - Discount factor of the mdp
    * threshold - Checks for convergence
'''

class ValueIteration(Environment):

    def __init__(self, grid_world, actions, rewards, gw, gh, gamma=0.99, epsilon=1e-4):

        '''
        Constuctor of super class
        '''

        super(ValueIteration, self).__init__(grid_world, actions, rewards, gw, gh)

        self.gamma = gamma

        '''
        Utility values of all states 
        Initialized to 0
        Aim of this class is to solve the utilities
        '''
        self.utilities = np.zeros((gh, gw))

        self.threshold = epsilon * (1 - self.gamma) / self.gamma

        self.policy = [[(1, 0) for _ in range(self.grid_width)] for _ in range(self.grid_height)]

        '''
        Data recovered during solving
        Required for analyis
        '''

        self.data = {}

        '''
        Initialize a list for every state
        '''

        for i in range(self.utilities.shape[0]):
            for j in range(self.utilities.shape[1]):
                cur_state = (i, j)
                self.data[f'{cur_state}'] = [0] 


    '''
    Solve the MDP
    '''

    def solve(self):
        iterations = 0
        while True:
            delta = 0
            iterations += 1
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
                    delta = max(delta, abs(new_utilities[i][j] - self.utilities[i][j]))
                    
                    self.data[f'{cur_state}'].append(state_value)

            self.utilities = new_utilities.copy()
            if delta < self.threshold:
                break

        self.policy = self.greedify(self.utilities)

        return self.utilities, iterations

    def greedify(self, utilities):
        policy = self.policy
        for i in range(utilities.shape[0]):
            for j in range(utilities.shape[1]):
                cur_state = (i, j)
                if self.is_wall(cur_state):
                        continue

                actions = self.actions
                action_values = {}
                for action in actions:
                    action_value = 0
                    transition_model = self.transition_model(cur_state, action)
                    for next_state in transition_model:
                        utility = utilities[next_state[0]][next_state[1]]
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
                policy[i][j] = best_action

        return policy
        
    def get_data(self):
        return self.data

    
