import numpy as np
import copy
import math
from ..environment.env import Environment


class PolicyIteration():

    """
    Policy Iteration Class

    Attributes
    ----------
    gamma : double
        Discount factor of the mdp
    k : int
        Number of Policy Evaluation updates
    data : dictionary
         Data to be used for analysis

    Methods
    ----------
    solve(mdp): Solves the mdp
    policy_evaluation(policy, utilities, mdp): Policy evaluation step
    policy_improvement(policy, utilities, mdp): Policy improvement step
    get_starting_policy(mdp): Initialize the starting policy of the algorithm
    get_data(): Return statistics
    """


    def __init__(self, gamma=0.99, k=100):
        """
        Parameters
        ----------
        gamma : double, optional
            Discount Factor of the algorithm (default is 0.99)
        k : int, optional
            Number of Policy Evaluation updates (default is 100)
        """

        self.gamma = gamma
        self.k = k

        self.data = {}


    def solve(self, mdp):
        """
        Solves the mdp by a series of policy evaluation and policy improvement steps

        Parameters
        ----------
        mdp : Markov Decision Process
            an MDP with states S, actions A(s), transition model P(s | s, a)

        Returns
        ----------
        utilities, policy and number of iterations
        """

        # Initialize the staring policy
        policy = self.get_starting_policy(mdp)

        # Initialize the starting utilities
        utilities = np.zeros((mdp.grid_height, mdp.grid_width), dtype=np.float)

        # Initialize the analysis data
        for i in range(utilities.shape[0]):
            for j in range(utilities.shape[1]):
                cur_state = (j, i)
                self.data[f'{cur_state}'] = [0] 

        # Initialize the total_iterations
        total_iterations = 0

        # Loop control variable
        is_policy_stable = False

        # Loop while the policy is not stable
        while not is_policy_stable:
            # Policy Evaluation
            utilities, iterations = self.policy_evaluation(policy, utilities, mdp)
            total_iterations += iterations
            
            # Policy Improvement
            policy, is_policy_stable = self.policy_improvement(policy, utilities, mdp)
        
        # Return utilities, policy and iterations as a dictionary
        return {"utilities": utilities, "policy": policy, "iterations": total_iterations}
    

    def policy_evaluation(self, policy, utilities, mdp):
        """
        Updates the utilities using the current policy

        Parameters
        ----------
        policy: 2D List
            Current policy

        utilities: 2D List
            Current utilities

        mdp : Environment object
            an MDP with states S, actions A(s), transition model P(s | s, a)
        

        Returns
        ----------
        utilities, iterations
        """

        iteration = 0

        # Loop control
        while iteration < self.k:
            iteration += 1

            # Copy current utilities
            new_utilities = utilities.copy()

            for i in range(utilities.shape[0]):
                for j in range(utilities.shape[1]):
                    cur_state = (i, j)

                    # Convert to (column, row) format
                    state_format = (cur_state[1], cur_state[0])

                    # Check if the current state is a wall
                    if mdp.is_wall(cur_state):
                        self.data[f'{state_format}'].append(0)
                        continue

                    # Get action from current policy
                    action = policy[i][j]

                    # Get transition model of the MDP
                    transition_model = mdp.transition_model(cur_state, action)

                    action_value = 0

                    # Loop through all the next states in the transition model
                    # This loop calculates expected utility for taking the action
                    for next_state in transition_model:
                        utility = utilities[next_state[0]][next_state[1]]
                        probability = transition_model[next_state]
                        expected_value = probability * utility
                        action_value += expected_value

                    # Reward of current state
                    reward = mdp.receive_reward(cur_state)

                    # Bellman Update
                    utility = reward + self.gamma * action_value
                    new_utilities[i][j] = utility

                    # Update analysis data
                    self.data[f'{state_format}'].append(utility)
            
            utilities = new_utilities.copy()

        return utilities, iteration

    def policy_improvement(self, policy, utilities, mdp):
        
        """
        Calculates the new policy using one step look ahead

        Parameters
        ----------
        policy: 2D List
            Current policy

        utilities: 2D List
            Current utilities

        mdp : Environment object
            an MDP with states S, actions A(s), transition model P(s | s, a)
        

        Returns
        ----------
        policy, is_stable
        """

        # Copy current policy
        new_policy = copy.deepcopy(policy)

        for i in range(utilities.shape[0]):
            for j in range(utilities.shape[1]):
                cur_state = (i, j)

                # Check if current state is a wall
                if mdp.is_wall(cur_state):
                        continue

                # Get all possible actions
                actions = mdp.actions
                action_values = {}

                # Loop through all possible actions
                # This loop calculates action values for all actions
                for action in actions:
                    action_value = 0

                    # Get transition model
                    transition_model = mdp.transition_model(cur_state, action)

                    # Loop through all the next states in the transition model
                    # This loop calculates expected utility for taking the action
                    for next_state in transition_model:
                        utility = utilities[next_state[0]][next_state[1]]
                        probability = transition_model[next_state]
                        expected_utility = probability * utility
                        action_value += expected_utility

                    # Set action value for the action
                    action_values[action] = action_value

                best_action = None
                best_action_value = -math.inf

                # This loop chooses the action that maximizes expected utility
                for action in action_values:
                    if action_values[action] > best_action_value:
                        best_action = action
                        best_action_value = action_values[action]
                new_policy[i][j] = best_action

        # Checks if the old policy is same as the new policy
        # If the old policy is same as the new policy, the policy is stable
        for i in range(len(policy)):
            for j in range(len(policy[i])):
                if policy[i][j] != new_policy[i][j]:
                    return new_policy, False
        
        return new_policy, True

    def get_starting_policy(self, mdp):
        """
        Starting policy for the algorithm. Current starting policy is to go NORTH at every step
        
        Parameters
        ----------
        mdp : Environment object
            an MDP with states S, actions A(s), transition model P(s | s, a)
        """

        # Initialize the starting policy
        policy = [[(-1, 0) for _ in range(mdp.grid_width)] for _ in range(mdp.grid_height)]
        return policy

    def get_data(self):
        """
        Returns data for analysis
        """

        return self.data
