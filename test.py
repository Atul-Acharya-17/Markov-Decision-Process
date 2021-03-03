#from mdp.environment.env import Environment
from mdp.algorithms.policy_iteration import PolicyIteration
from mdp.algorithms.value_iteration import ValueIteration
from file_manager import FileManager


grid_world = [
    ['','W','','','', ''],
    ['','','','','W', ''],
    ['','','','','', ''],
    ['','','','','', ''],
    ['','W','W','W','', ''],
    ['','','','','', ''],
]

rewards = [
    [1, 0, 1, -0.04, -0.04, 1],
    [-0.04,-1,-0.04,1,0, -1],
    [-0.04,-0.04,-1,-0.04,1, -0.04],
    [-0.04,-0.04,-0.04,-1,-0.04, 1],
    [-0.04,0,0,0,-1, -0.04],
    [-0.04,-0.04,-0.04,-0.04,-0.04, -0.04],
]

actions = [(1,0), (0,1), (-1,0), (0,-1)]

gw = 6
gh=6

pi = ValueIteration(grid_world, actions, rewards, gw, gh, 0.99)

values = pi.solve(1000, 0.01)

for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        print(f"{j, i}: {values[j][i]}")

file_manager = FileManager('analysis/')
file_manager.write('pi.csv', pi.get_data())