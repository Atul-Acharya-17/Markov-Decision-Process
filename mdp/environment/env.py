import numpy as np


class Environment():

    """
    Environment Class
    Stores information about the MDP

    Attributes
    ----------
    grid_world : 2-D List
        Stores information about walls
    actions : List
        All possible actions
    rewards : 2-D List
        Rewards for each state
    grid_width : int
        Width of the grid
    grid_height : int
         Height of the grid

    Methods
    ----------
    receive_reward(state): Returns the reward for the state
    get_actions(): Returns the list of all actions
    transition_model(state, action): Returns the transition model P(s'|s,a)
    is_wall(state): Checks whether the state is wall
    """

    def __init__(self, grid_world, actions, rewards, grid_width, grid_height):

        """
        Parameters
        ----------
        grid_world : 2-D List
            Grid World of the MDP
        
        actions : List
            List of all possible actions
        
        rewards : 2-D List
            Rewards for each state

        grid_width : int
            Width of the grid
        grid_height : int
            Height of the grid
        """

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.actions = actions
        self.num_actions = len(actions)
        self.rewards = rewards
        self.grid_world = grid_world


    def receive_reward(self, state):
        """
        Returns the reward for the state

        Parameters
        ----------
        state : Tuple
            State to calculate the reward for
        """
        return self.rewards[state[0]][state[1]]


    def get_actions(self):
        """
        Returns list of all actions
        """
        return self.actions


    def transition_model(self, state, action):

        """
        Returns the transition model P(s'| s, a)

        Parameters
        ----------
        state : Tuple
            Current State - s 
        
        action : Tuple
            Action to take - a 
        """    

        model = {}
        
        """
        The intended outcome occurs with probability 0.8, and
        with probability 0.1 the agent moves at either right angle to the intended direction. If the
        move would make the agent walk into a wall, the agent stays in the same place as before.

        The actions are represented by the following tuple
        NORTH : (-1, 0)
        SOUTH : (1, 0)
        EAST  :  (0, 1)
        WEST  :  (0, -1)

        possible_directions = [action, (action[1], action[0]), (-action[1], -action[0])]

        Evaluating this for each action gives:

        NORTH : (-1, 0), (0, -1), (0, 1) --> NORTH, WEST, EAST
        SOUTH : (1, 0), (0, 1), (0, -1)  --> SOUTH, EAST, WEST
        EAST  : (0, 1), (1, 0), (-1, 0)  --> EAST, SOUTH, NORTH
        WEST  : (0, -1), (-1, 0), (1, 0) --> WEST, NORTH, SOUTH
        """
        possible_directions = [action, (action[1], action[0]), (-action[1], -action[0])]

        # Probability for going in the same direction is 0.8, probability for going in each perpendicular direction is 0.1
        probability = [0.8, 0.1, 0.1]

        # Loop through all possible directions the agent can go to by taking the action
        for idx, direction in enumerate(possible_directions):

            # Calculate next state
            next_state = (state[0] + direction[0], state[1] + direction[1])

            # Check if next state is within the boundaries of the grid
            if 0 <= next_state[0] < self.grid_width and 0 <= next_state[1] < self.grid_height:

                # Check if next state is a wall
                if self.grid_world[next_state[0]][next_state[1]] == 'W':
                    next_state = state
            else:
                next_state = state
            
            # Check if next state is already in the transition model
            # If true then add the probabilities
            if next_state in model:
                model[next_state] += probability[idx]
            else:
                model[next_state] = probability[idx]

        return model

    def is_wall(self, state):

        """
        Check if the state is a wall
        """

        return self.grid_world[state[0]][state[1]] == 'W'
