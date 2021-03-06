from mdp.algorithms.value_iteration import ValueIteration
from file_manager import FileManager

from config import grid, actions, rewards, gw, gh


'''
GAMMA and THRESHOLD values
'''

GAMMA = 0.99
THRESHOLD = 1e-4
PATH = 'analysis/'

'''
Initialize the ValueIteration object and solve the MDP 
'''
value_iteration = ValueIteration(grid, actions, rewards, gw, gh, GAMMA)

'''
The solve method returns the utilities of all the states in a grid like format and the number of iterations
'''
values, num_iterations = value_iteration.solve(THRESHOLD)


'''
Print the details to the console
'''
print(f'Number of iterations: {num_iterations}\n')

for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        print(f"{i, j}: {values[j][i]}")


'''
Write the data to a file
'''
file_mgr = FileManager(PATH)
file_mgr.write('value_iteration.csv', value_iteration.get_data())