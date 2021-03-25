from mdp.algorithms.value_iteration import ValueIteration
from mdp.environment.env import Environment
from file_manager import FileManager
import pygame

# Change file name to custom_grid to use your own grid
from config import grid, actions, rewards, gw, gh


pygame.init()

# Initialize Constants
GAMMA = 0.99
C = 0.1
MAX_REWARD = 1.0
EPSILON = C * MAX_REWARD
PATH = 'analysis/'
CONVERT_POLICY = {(1,0): '↓', (-1, 0): '↑' , (0, 1): '→', (0, -1): '←'}
DISPLAY_GRID = True

# Initialize the MDP
mdp = Environment(grid, actions, rewards, gw, gh)

# Initialize the algorithm
value_iteration = ValueIteration(GAMMA)

# Solve the MDP
results = value_iteration.solve(mdp, EPSILON)

# Retrieve the results
num_iterations = results['iterations']
values = results['utilities']
policy = results['policy']

# Print the results to the console
print(f'Number of iterations: {num_iterations}\n')
print('\n(Column, Row)')
for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        print(f"{j, i}: {values[j][i]}")

# Save data for analysis
file_mgr = FileManager(PATH)
file_mgr.write('value_iteration.csv', value_iteration.get_data())

# Display utility and policy plot
if DISPLAY_GRID:

    GREEN = (100, 200, 100)
    RED = (200, 100, 100)
    WHITE = (200, 200, 200)
    GREY = (50, 50, 50)

    directions = [[CONVERT_POLICY[cell] for cell in row] for row in policy]
    utilities = [["{:.3f}".format(cell) for cell in row] for row in values]

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
    policy_font = pygame.font.Font("assets/seguisym.ttf", 30)
    utility_font = pygame.font.Font("assets/seguisym.ttf", 15)

    screen = pygame.display.set_mode(screen_dimensions)
    pygame.display.set_caption('Policy Iteration')

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
                screen.blit(message, (col * block_size + 17, row * block_size + 5))

        pygame.display.update()
        #pygame.image.save(screen, "images/value_iteration/policy.png")
    

    screen = pygame.display.set_mode(screen_dimensions)
    pygame.display.set_caption('Policy Iteration')

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
                screen.blit(message, (col * block_size + 4, row * block_size + 14))

        pygame.display.update()
        #pygame.image.save(screen, "images/value_iteration/values.png")