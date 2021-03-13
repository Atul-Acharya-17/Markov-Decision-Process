from ..environment.env import Environment
import numpy as np
import math


class ValueIteration():

    """
    Value Iteration Class

    Attributes
    ----------
    gamma : double 
        Discount factor of the mdp
    data : dictionary
        Data to be used for analysis

    Methods
    ----------
    solve(mdp, epsilon): Solves mdp
    greedify(utilities, mdp): Calculates the optimal policy by selecting the greedy action
    get_data(): Returns statistics
    """

    def __init__(self, gamma=0.99):

        """
        Parameters
        ----------
        gamma : double, optional
            Discount Factor of the algorithm (default is 0.99)
        """

        self.gamma = gamma

        self.data = {}


    def solve(self, mdp, epsilon):
        """
        Solve the MDP

        Parameters
        ----------

        mdp : Environment object
            MDP to solve
        
        epsilon : double
            Maximum error allowed in the utility of any state
        """
        
        # Initialize the utilities to 0
        utilities = np.zeros((mdp.grid_height, mdp.grid_width), dtype=np.float)

        # Initialize the analysis data
        for i in range(utilities.shape[0]):
            for j in range(utilities.shape[1]):
                cur_state = (i, j)
                self.data[f'{cur_state}'] = [0]

        # Calculate the threshold
        threshold = epsilon * (1 - self.gamma) / self.gamma
        iterations = 0

        while True:
            delta = 0
            iterations += 1

            # Copy the current utilities
            new_utilities = utilities.copy()

            # Loop through all the states
            for i in range(utilities.shape[0]):
                for j in range(utilities.shape[1]):
                    cur_state = (i, j)

                    # Check if the current state is a wall
                    if mdp.is_wall(cur_state):
                        self.data[f'{cur_state}'].append(0)
                        continue

                    # Get all possible actions
                    actions = mdp.get_actions()
                    
                    action_values = []

                    # Loop through all the actions
                    # This loop calculates the action values for each action
                    for action in actions:

                        action_value = 0

                        # Get the transition model
                        transition_model = mdp.transition_model(cur_state, action)

                        # Loop through all the next states in the transition model
                        # This loop calculates expected utility for taking the action
                        for next_state in transition_model:
                            utility = utilities[next_state[0]][next_state[1]]
                            probability = transition_model[next_state]
                            expected_utility = probability * utility
                            action_value += expected_utility
                        
                        action_values.append(action_value)

                    # Reward for the current state
                    reward = mdp.receive_reward(cur_state)

                    # Bellman Update
                    state_value = reward + self.gamma * max(action_values)
                    new_utilities[i][j] = state_value

                    # Update the value of delta
                    delta = max(delta, abs(new_utilities[i][j] - utilities[i][j]))
                    
                    # Update the analysis data
                    self.data[f'{cur_state}'].append(state_value)

            # Update the utilities
            utilities = new_utilities.copy()

            # Check for convergence
            if delta < threshold:
                break

        # Calculate optimal policy
        policy = self.greedify(utilities, mdp)

        # Return results
        return {"utilities": utilities, "policy": policy, "iterations":iterations}

    def greedify(self, utilities, mdp):
        
        """
        Find optimal policy by taking greedy actions

        Parameters
        ----------

        utilities : 2-D List
            Utilities of each state
        
        mdp : Environment object
            MDP to solve
        """

        policy = [[(1, 0) for _ in range(utilities.shape[0])] for _ in range(utilities.shape[1])]
        for i in range(utilities.shape[0]):
            for j in range(utilities.shape[1]):
                cur_state = (i, j)

                # Check if current state is a wall
                if mdp.is_wall(cur_state):
                        continue

                actions = mdp.actions
                action_values = {}

                # Loop through all actions
                for action in actions:
                    action_value = 0
                    # Get transition model
                    transition_model = mdp.transition_model(cur_state, action)

                    # Loop through all the next states in the transition model
                    # Calculate the expected utility for taking the action
                    for next_state in transition_model:
                        utility = utilities[next_state[0]][next_state[1]]
                        probability = transition_model[next_state]
                        expected_utility = probability * utility
                        action_value += expected_utility
                    action_values[action] = action_value
                best_action = None
                best_action_value = -math.inf

                # Choose best action
                for action in action_values:
                    if action_values[action] > best_action_value:
                        best_action = action
                        best_action_value = action_values[action]
                policy[i][j] = best_action

        return policy
        
    def get_data(self):
        """
        Returns data for analysis
        """

        return self.data
