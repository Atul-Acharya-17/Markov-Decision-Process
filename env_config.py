################################################################
#   Environment of the assignment
################################################################


################################################################
#   Width and Height of grid world
#
#   gw: width of grid world
#   gh: height of grid world
################################################################

gw = 6
gh = 6

################################################################
#   Grid World
#   Feel free to modify the grid world in any way
################################################################

grid = [
    ['G','W','G','','', 'G'],
    ['','R','','G','W', 'R'],
    ['','','R','','G', ''],
    ['','','','R','', 'G'],
    ['','W','W','W','R', ''],
    ['','','','','', ''],
]


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

actions = [
    (-1, 0), 
    (1, 0), 
    (0, 1), 
    (0, -1)
]
num_actions = len(actions)