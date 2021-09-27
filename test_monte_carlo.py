from mdp.algorithms.monte_carlo import MonteCarlo
from mdp.environment.env import Environment
import numpy as np

import pygame

pygame.init()

from env_config import grid, actions, rewards, gw, gh

CONVERT_POLICY = {1: '↓', 0: '↑' , 2: '→', 3: '←'}
DISPLAY_GRID = True

UTILITY_FONT_SIZE = 15
UTILITY_OFFSET = (4, 14)

POLICY_FONT_SIZE = 30
POLICY_OFFSET = (17, 5)

ratio = 1

mdp = Environment(grid, actions, rewards, gw, gh)

q_learning = MonteCarlo(n_w=6, n_h=6, n_actions=4)

q_table = q_learning.solve(mdp=mdp)

print('\n(Column, Row)')
for i in range(q_table.shape[0]):
    for j in range(q_table.shape[1]):
        print(f"{j, i}: {max(q_table[i][j])}")

# Display utility and policy plot
if DISPLAY_GRID:

    GREEN = (100, 200, 100)
    RED = (200, 100, 100)
    WHITE = (200, 200, 200)
    GREY = (50, 50, 50)

    directions = [[CONVERT_POLICY[np.argmax(cell)] for cell in row] for row in q_table]
    utilities = [["{:.3f}".format(np.max(cell)) for cell in row] for row in q_table]

    colors = []
    for row in grid:
        color = []
        for cell in row:
            if cell == 'W':
               color.append(GREY)
            elif cell == 'G':
                color.append(GREEN)
            elif cell == 'R':
                color.append(RED)
            else:
                color.append(WHITE)
        colors.append(color)
    
    block_size = 50
    width = 300
    height = 300
    screen_dimensions = (width, height)
    screen_color = (0, 0, 0)
    policy_font = pygame.font.Font("assets/seguisym.ttf", int(POLICY_FONT_SIZE*ratio))
    utility_font = pygame.font.Font("assets/seguisym.ttf", int(UTILITY_FONT_SIZE*ratio))

    screen = pygame.display.set_mode(screen_dimensions)
    pygame.display.set_caption('Monte Carlo')

    # Display Policy
    running = True
    while running:

        for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

        rect = pygame.Rect(0, 0, width, height)
        pygame.draw.rect(screen, screen_color, rect)

        for row in range(len(grid)):
            for col in range(len(grid)):
                rect = pygame.Rect(col * block_size, row * block_size, block_size, block_size)
                pygame.draw.rect(screen, colors[row][col], rect)
                pygame.draw.rect(screen, (0, 0, 0), rect, 1)

                if grid[row][col] == 'W':
                    continue
                message = policy_font.render(directions[row][col], True, (0, 0, 0))
                screen.blit(message, (col * block_size + POLICY_OFFSET[0] * ratio, row * block_size + POLICY_OFFSET[1]*ratio))

        pygame.display.update()

    screen = pygame.display.set_mode(screen_dimensions)
    pygame.display.set_caption('Monte Carlo')

    # Display Utilities
    running = True
    while running:

        for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

        rect = pygame.Rect(0, 0, width, height)
        pygame.draw.rect(screen, screen_color, rect)

        for row in range(len(grid)):
            for col in range(len(grid)):
                rect = pygame.Rect(col * block_size, row * block_size, block_size, block_size)
                pygame.draw.rect(screen, colors[row][col], rect)
                pygame.draw.rect(screen, (0, 0, 0), rect, 1)

                if grid[row][col] == 'W':
                    continue
                message = utility_font.render(utilities[row][col], True, (0, 0, 0))
                screen.blit(message, (col * block_size + UTILITY_OFFSET[0]*ratio, row * block_size + UTILITY_OFFSET[1]*ratio))

        pygame.display.update()