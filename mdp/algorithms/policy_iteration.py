import numpy as np
import copy
import math
from ..environment.env import Environment


'''
Policy Iteration Class
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

class PolicyIteration(Environment):

    def __init__(self, grid_world, actions, rewards, gw, gh, gamma=0.99, threshold=1e-4):\
        
        '''
        Constuctor of super class
        '''

        super(PolicyIteration, self).__init__(grid_world, actions, rewards, gw, gh)

        self.gamma = gamma

        '''
        Initial policy that the algorithm will be using
        '''
        self.policy = self.get_starting_policy()

        '''
        Utility values of all states 
        Initialized to 0
        Aim of this class is to solve the utilities
        '''

        self.utilities = np.zeros((gh, gw))

        self.threshold = threshold

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
    
    While th policy is not state, the algorithm will run a series
    of policy evaluation and policy improvement steps
    '''

    def solve(self):
        iterations = 0
        stable_policy = False
        while not stable_policy:
            iterations += 1
            self.policy_evaluation()
            stable_policy = self.policy_improvement()
        return self.utilities, iterations
    
    '''
    Policy evaluation 
    '''

    def policy_evaluation(self):
        while True:
            new_utilities = self.utilities.copy()
            delta = 0
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
                    delta = max(delta, abs(new_utilities[i][j] - self.utilities[i][j]))
                    self.data[f'{cur_state}'].append(utility)
            
            self.utilities = new_utilities.copy()
            if delta < self.threshold:
                break

    '''
    Policy improvement
    '''
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

    
    '''
    Starting policy for the algorithm

    Current starting policy is to go SOUTH at every step
    '''
    def get_starting_policy(self):
        policy = [[(1, 0) for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        return policy


    def get_data(self):
        return self.data

    

