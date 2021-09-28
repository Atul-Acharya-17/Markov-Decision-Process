import pygame
pygame.init()
# Display Settings

UTILITY_FONT_SIZE = 15
UTILITY_OFFSET = (4, 14)
POLICY_FONT_SIZE = 30
POLICY_OFFSET = (17, 5)
RATIO = 1
POLICY_FONT = pygame.font.Font("assets/seguisym.ttf", int(POLICY_FONT_SIZE * RATIO))
UTILITY_FONT = pygame.font.Font("assets/seguisym.ttf", int(UTILITY_FONT_SIZE * RATIO))

block_size=50
width=300
height=300

# Value Iteration
vi_gamma = 0.99
vi_c = 0.1
MAX_REWARD = 1
vi_epsilon = vi_c * MAX_REWARD


# Policy Iteration
pi_gamma = 0.99
pi_k = 100

# Q Learning
q_learning_gamma = 0.99
q_learning_step_size = 0.1
q_learning_num_episodes = 50000
q_learning_num_steps = 100
q_learning_epsilon = 0.25

# Expected SARSA
expected_sarsa_gamma = 0.99
expected_sarsa_step_size = 0.1 
expected_sarsa_num_episodes = 50000 
expected_sarsa_num_steps = 100
expected_sarsa_epsilon = 0.1

# SARSA
sarsa_gamma = 0.99
sarsa_step_size = 0.1 
sarsa_num_episodes = 50000 
sarsa_num_steps = 100
sarsa_epsilon = 0.1

# Monte Carlo
monte_carlo_gamma = 0.99
monte_carlo_num_episodes = 100000 
monte_carlo_num_steps = 1000
monte_carlo_epsilon = 0.1

# Dyna Q

dyna_q_gamma = 0.99
dyna_q_step_size = 0.1
dyna_q_num_episodes = 50000
dyna_q_num_steps = 100
dyna_q_epsilon = 0.25
dyna_q_planning_steps = 100