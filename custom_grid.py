import numpy as np

np.random.seed(0)

################################################################
#   Create your environment in this file
################################################################


################################################################
#   Width and Height of grid world
#
#   gw: width of grid world
#   gh: height of grid world
################################################################

gw = 20
gh = 20

################################################################
#   Grid World
#   Feel free to modify the grid world in any way
################################################################

probability = {'G': 0.166, 'R': 0.33, 'W': 0.5, '': 1.0}

prob = np.random.rand(gh, gw)

grid = []

for row in prob:
    grid_row = []
    for cell in row:
        if cell <= 0.166:
            grid_row.append('G')
        elif cell <= 0.33:
            grid_row.append('R')
        elif cell <= 0.5:
            grid_row.append('W')
        else:
            grid_row.append('')
    grid.append(grid_row)

################################################################
#   Rewards
#   Feel free to modify the rewards in any way
#   Make sure the reward array has the same shape
#   as the grid array
################################################################

reward_map = {'G': +1, 'R': -1, 'W': 0, '': -0.04}

rewards = [[reward_map[cell] for cell in row] for row in grid]

################################################################
#   Actions
#   The actions are NORTH, SOUTH, EAST and WEST
#   Each action is represented by a tuple
#   
#   NORTH: (-1, 0)
#   SOUTH: (1, 0)
#   EAST:  (0, 1)
#   WEST:  (0, -1)
################################################################

actions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
num_actions = len(actions)