import numpy as np


class Environment():

    '''
    Environment Class
    Stores information about the MDP

    Attributes
    ----------
    grid_world
    actions
    rewards
    grid_width
    grid_height

    Methods
    ----------

    '''

    def __init__(self, grid_world, actions, rewards, grid_width, grid_height):

        self.grid_width = grid_width
        self.grid_height = grid_height
       
        '''
        List of all possible actions
        '''
        self.actions = actions
        self.num_actions = len(actions)

        '''
        Rewards for each state
        '''
        self.rewards = rewards

        self.grid_world = grid_world


    '''
    Returns the reward for a particular state
    '''
    def receive_reward(self, state):
        return self.rewards[state[0]][state[1]]



    '''
    Returns list of all actions
    '''
    def get_actions(self):
        return self.actions


    def transition_model(self, state, action):
        model = {}
        possible_directions = [action, (action[1], action[0]), (-action[1], -action[0])]
        probability = [0.8, 0.1, 0.1]

        for idx, direction in enumerate(possible_directions):
            next_state = (state[0] + direction[0], state[1] + direction[1])
            if 0 <= next_state[0] < self.grid_width and 0 <= next_state[1] < self.grid_height:
                if self.grid_world[next_state[0]][next_state[1]] == 'W':
                    next_state = state
            else:
                #continue
                next_state = state

            if next_state in model:
                model[next_state] += probability[idx]
            else:
                model[next_state] = {}
                model[next_state] = probability[idx]

        return model

    def is_wall(self, state):
        return self.grid_world[state[0]][state[1]] == 'W'
