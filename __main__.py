import numpy as np

from mdp.environment.env import Environment
from mdp.algorithms.value_iteration import ValueIteration
from mdp.algorithms.policy_iteration import PolicyIteration
from mdp.algorithms.monte_carlo import MonteCarlo
from mdp.algorithms.sarsa import SARSA
from mdp.algorithms.expected_sarsa import ExpectedSarsa
from mdp.algorithms.q_learning import QLearning
from mdp.algorithms.dyna_q import DynaQ

from display_manager import DisplayManager

from config import *

# Change file name to custom_grid to use your own grid
from env_config import grid, actions, rewards, gw, gh

import argparse

CONVERT_POLICY_TUPLE = {(1,0): '↓', (-1, 0): '↑' , (0, 1): '→', (0, -1): '←'}
CONVERT_POLICY_INT= {1: '↓', 0: '↑' , 2: '→', 3: '←'}

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep Colorizer")
    parser.add_argument(
        "--algorithm", help="Select Algorithm",
        default="q_learning", required=True)
    parser.add_argument(
        "--display_policy", help="Display image of policy", default=False,
        required=False)
    parser.add_argument(
        "--display_utilities", help="Display image of utilities", default=False,
        required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    mdp = Environment(grid, actions, rewards, gw, gh)
    algorithm = args.algorithm

    if algorithm == 'value_iteration':

        value_iteration = ValueIteration(gamma=vi_gamma)
        results = value_iteration.solve(mdp, epsilon=vi_epsilon)

        num_iterations = results['iterations']
        values = results['utilities']
        policy = results['policy']

        print(f'Number of iterations: {num_iterations}\n')
        print('\n(Column, Row)')
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                print(f"{j, i}: {values[i][j]}")

        if args.display_policy:
            directions = [[CONVERT_POLICY_TUPLE[cell] for cell in row] for row in policy]
            
            display_manager = DisplayManager(block_size=block_size, width=width, height=height)

            display_manager.display(array=directions, grid=grid, offset=POLICY_OFFSET, font=POLICY_FONT, title='Value Iteration')

        if args.display_utilities:
            utilities = [["{:.3f}".format(cell) for cell in row] for row in values]
            
            display_manager = DisplayManager(block_size=block_size, width=width, height=height)

            display_manager.display(array=utilities, grid=grid, offset=UTILITY_OFFSET, font=UTILITY_FONT, title='Value Iteration')


    elif algorithm == 'policy_iteration':
        policy_iteration = PolicyIteration(gamma=pi_gamma, k=pi_k)
        results = policy_iteration.solve(mdp)

        num_iterations = results['iterations']
        values = results['utilities']
        policy = results['policy']

        print(f'Number of iterations: {num_iterations}\n')
        print('\n(Column, Row)')
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                print(f"{j, i}: {values[i][j]}")

        if args.display_policy:
            directions = [[CONVERT_POLICY_TUPLE[cell] for cell in row] for row in policy]
            
            display_manager = DisplayManager(block_size=block_size, width=width, height=height)

            display_manager.display(array=directions, grid=grid, offset=POLICY_OFFSET, font=POLICY_FONT, title='Policy Iteration')

        if args.display_utilities:
            utilities = [["{:.3f}".format(cell) for cell in row] for row in values]
            
            display_manager = DisplayManager(block_size=block_size, width=width, height=height)

            display_manager.display(array=utilities, grid=grid, offset=UTILITY_OFFSET, font=UTILITY_FONT, title='Policy Iteration')

    elif algorithm == 'sarsa':
        sarsa = SARSA(n_w=mdp.grid_width, n_h=mdp.grid_height, n_actions=mdp.num_actions, gamma=sarsa_gamma, epsilon=sarsa_epsilon,
                num_episodes=sarsa_num_episodes, num_steps=sarsa_num_steps, step_size=sarsa_step_size)

        q_table = sarsa.solve(mdp=mdp)

        print('\n(Column, Row)')
        for i in range(q_table.shape[0]):
            for j in range(q_table.shape[1]):
                print(f"{j, i}: {max(q_table[i][j])}")        

        if args.display_policy:
            directions = [[CONVERT_POLICY_INT[np.argmax(cell)] for cell in row] for row in q_table]
            
            display_manager = DisplayManager(block_size=block_size, width=width, height=height)

            display_manager.display(array=directions, grid=grid, offset=POLICY_OFFSET, font=POLICY_FONT, title='SARSA')

        if args.display_utilities:
            utilities = [["{:.3f}".format(np.max(cell)) for cell in row] for row in q_table]
            
            display_manager = DisplayManager(block_size=block_size, width=width, height=height)

            display_manager.display(array=utilities, grid=grid, offset=UTILITY_OFFSET, font=UTILITY_FONT, title='SARSA')

    elif algorithm == 'expected_sarsa':
        expected_sarsa = ExpectedSarsa(n_w=mdp.grid_width, n_h=mdp.grid_height, n_actions=mdp.num_actions, gamma=expected_sarsa_gamma, epsilon=expected_sarsa_epsilon,
        num_episodes=expected_sarsa_num_episodes, num_steps=expected_sarsa_num_steps, step_size=expected_sarsa_step_size)

        q_table = expected_sarsa.solve(mdp=mdp)

        print('\n(Column, Row)')
        for i in range(q_table.shape[0]):
            for j in range(q_table.shape[1]):
                print(f"{j, i}: {max(q_table[i][j])}")    

        if args.display_policy:
            directions = [[CONVERT_POLICY_INT[np.argmax(cell)] for cell in row] for row in q_table]
            
            display_manager = DisplayManager(block_size=block_size, width=width, height=height)

            display_manager.display(array=directions, grid=grid, offset=POLICY_OFFSET, font=POLICY_FONT, title='Expected SARSA')

        if args.display_utilities:
            utilities = [["{:.3f}".format(np.max(cell)) for cell in row] for row in q_table]
            
            display_manager = DisplayManager(block_size=block_size, width=width, height=height)

            display_manager.display(array=utilities, grid=grid, offset=UTILITY_OFFSET, font=UTILITY_FONT, title='Expected SARSA')

    elif algorithm == 'q_learning':
        q_learning = QLearning(n_w=mdp.grid_width, n_h=mdp.grid_height, n_actions=mdp.num_actions, gamma=q_learning_gamma, epsilon=q_learning_epsilon,
        num_episodes=q_learning_num_episodes, num_steps=q_learning_num_steps, step_size=q_learning_step_size)

        q_table = q_learning.solve(mdp=mdp)

        print('\n(Column, Row)')
        for i in range(q_table.shape[0]):
            for j in range(q_table.shape[1]):
                print(f"{j, i}: {max(q_table[i][j])}") 

        if args.display_policy:
            directions = [[CONVERT_POLICY_INT[np.argmax(cell)] for cell in row] for row in q_table]
            
            display_manager = DisplayManager(block_size=block_size, width=width, height=height)

            display_manager.display(array=directions, grid=grid, offset=POLICY_OFFSET, font=POLICY_FONT, title='Q Learning')

        if args.display_utilities:
            utilities = [["{:.3f}".format(np.max(cell)) for cell in row] for row in q_table]
            
            display_manager = DisplayManager(block_size=block_size, width=width, height=height)

            display_manager.display(array=utilities, grid=grid, offset=UTILITY_OFFSET, font=UTILITY_FONT, title='Q Learning')  

            
    elif algorithm == 'monte_carlo':
        monte_carlo = MonteCarlo(n_w=mdp.grid_width, n_h=mdp.grid_height, n_actions=mdp.num_actions, gamma=monte_carlo_gamma, epsilon=monte_carlo_epsilon,
        num_episodes=monte_carlo_num_episodes, num_steps=monte_carlo_num_steps)
        q_table = monte_carlo.solve(mdp=mdp)

        print('\n(Column, Row)')
        for i in range(q_table.shape[0]):
            for j in range(q_table.shape[1]):
                print(f"{j, i}: {max(q_table[i][j])}")  

        if args.display_policy:
            directions = [[CONVERT_POLICY_INT[np.argmax(cell)] for cell in row] for row in q_table]
            
            display_manager = DisplayManager(block_size=block_size, width=width, height=height)

            display_manager.display(array=directions, grid=grid, offset=POLICY_OFFSET, font=POLICY_FONT, title='Monte Carlo')

        if args.display_utilities:
            utilities = [["{:.3f}".format(np.max(cell)) for cell in row] for row in q_table]
            
            display_manager = DisplayManager(block_size=block_size, width=width, height=height)

            display_manager.display(array=utilities, grid=grid, offset=UTILITY_OFFSET, font=UTILITY_FONT, title='Monte Carlo')  

    elif algorithm == 'dyna_q':
        dyna_q = DynaQ(n_w=mdp.grid_width, n_h=mdp.grid_height, n_actions=mdp.num_actions, gamma=dyna_q_gamma, epsilon=dyna_q_epsilon,
        num_episodes=dyna_q_num_episodes, num_steps=dyna_q_num_steps, step_size=dyna_q_step_size, planning_steps=dyna_q_planning_steps)
    
        q_table = dyna_q.solve(mdp=mdp)

        print('\n(Column, Row)')
        for i in range(q_table.shape[0]):
            for j in range(q_table.shape[1]):
                print(f"{j, i}: {max(q_table[i][j])}")    

        if args.display_policy:
            directions = [[CONVERT_POLICY_INT[np.argmax(cell)] for cell in row] for row in q_table]
            
            display_manager = DisplayManager(block_size=block_size, width=width, height=height)

            display_manager.display(array=directions, grid=grid, offset=POLICY_OFFSET, font=POLICY_FONT, title='Dyna Q')

        if args.display_utilities:
            utilities = [["{:.3f}".format(np.max(cell)) for cell in row] for row in q_table]
            
            display_manager = DisplayManager(block_size=block_size, width=width, height=height)

            display_manager.display(array=utilities, grid=grid, offset=UTILITY_OFFSET, font=UTILITY_FONT, title='Dyna Q')

    else:
        print("Invalid Choice")
        print("The following options for algorithm are:")
        print("value_iteration\npolicy_iteration\nsarsa\nexpected_sarsa\nq_learning\nmonte_carlo\ndyna_q")