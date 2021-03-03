import numpy as np


class Environment():

    # OR MAKE EVERYTHING IN DICTIONARIES
    def __init__(self, grid_world, actions, rewards, grid_width, grid_height):

        # grid size along 1 dimension
        self.grid_width = grid_width
        self.grid_height = grid_height
        # 1-D numpy array of [0, 1, 2, 3] where 1: North, 2: East, 3:South, 4: West
        self.actions = actions
        self.num_actions = len(actions)

        # 2D numpy array or dictionary with list as key and reward as value
        self.rewards = rewards

        self.grid_world = grid_world


    # self.actions = [(1, 0), (0, 1), ( -1, 0), (0, -1)]

    def receive_reward(self, state):
        return self.rewards[state[0]][state[1]]

    def transition_model(self, state, action):
        model = {}
        for action in self.actions:
            possible_directions = [action, (action[1], action[0]), (-action[1], -action[0])]
            probability = [0.8, 0.1, 0.1]

            for idx, direction in enumerate(possible_directions):
                next_state = (state[0] + direction[0], state[1] + direction[1])
                if 0 <= next_state[0] < self.grid_width and 0 <= next_state[1] < self.grid_height:
                    if self.grid_world[next_state[0]][next_state[1]] == 'W':
                        next_state = state
                else:
                    next_state = state

                if action in model:
                    if next_state not in model[action]:
                        model[action][next_state] = probability[idx]
                    else:
                        model[action][next_state] += probability[idx]
                else:
                    model[action] = {}
                    model[action][next_state] = probability[idx]

        return model


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
