from mdp.algorithms.value_iteration import ValueIteration
from file_manager import FileManager
import pygame
from config import grid, actions, rewards, gw, gh


pygame.init()


'''
GAMMA and THRESHOLD values
'''

GAMMA = 0.99
C = 60
MAX_REWARD = 1.0
EPSILON = C * MAX_REWARD
PATH = 'analysis/'


CONVERT_POLICY = {(1,0): '↓', (-1, 0): '↑' , (0, 1): '→', (0, -1): '←'}
DISPLAY_GRID = True

'''
Initialize the ValueIteration object and solve the MDP 
'''
value_iteration = ValueIteration(grid, actions, rewards, gw, gh, GAMMA, EPSILON)

'''
The solve method returns the utilities of all the states in a grid like format and the number of iterations
'''
values, num_iterations = value_iteration.solve()


'''
Print the details to the console
'''
print(f'Number of iterations: {num_iterations}\n')

for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        print(f"{i, j}: {values[i][j]}")


'''
Write the data to a file
'''
file_mgr = FileManager(PATH)
file_mgr.write('value_iteration.csv', value_iteration.get_data())


if DISPLAY_GRID:


    GREEN = (100, 200, 100)
    RED = (200, 100, 100)
    WHITE = (200, 200, 200)
    GREY = (50, 50, 50)


    policy = value_iteration.policy

    directions = [[CONVERT_POLICY[_] for _ in row] for row in policy]#convert_policy()
    # Get rewards to color the squares

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
    screen = pygame.display.set_mode(screen_dimensions)
        # color of the screen
    screen_color = (0, 0, 0)

    font = pygame.font.Font("assets/seguisym.ttf", 30)

    running = True

    pygame.display.set_caption('Value Iteration')

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
                message = font.render(directions[row][col], True, (0, 0, 0))
                screen.blit(message, (col * block_size + 17, row * block_size + 5))

        pygame.display.update()