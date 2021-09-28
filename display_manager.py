import pygame
pygame.init()

class DisplayManager(object):

    GREEN = (100, 200, 100)
    RED = (200, 100, 100)
    WHITE = (200, 200, 200)
    GREY = (50, 50, 50)
    SCREEN_COLOR = (0, 0, 0)

    RATIO = 1

    def __init__(self, block_size=50, width=300, height=300) -> None:
        super().__init__()
        self.block_size = block_size
        self.width = width
        self.height = height
        self.screen_dimensions = (width, height)

    def get_colors_for_grid(self, grid):
        colors = []
        for row in grid:
            color = []
            for cell in row:
                if cell == 'W':
                    color.append(DisplayManager.GREY)
                elif cell == 'G':
                    color.append(DisplayManager.GREEN)
                elif cell == 'R':
                    color.append(DisplayManager.RED)
                else:
                    color.append(DisplayManager.WHITE)
            colors.append(color)
        return colors

    def display(self, array, grid, offset, font, title='Plot', save=False, file_name='image.png'):
        screen = pygame.display.set_mode(self.screen_dimensions)
        pygame.display.set_caption(title)

        colors = self.get_colors_for_grid(grid=grid)
        running = True
        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            rect = pygame.Rect(0, 0, self.width, self.height)
            pygame.draw.rect(screen, DisplayManager.SCREEN_COLOR, rect)

            for row in range(len(grid)):
                for col in range(len(grid)):
                    rect = pygame.Rect(col * self.block_size, row * self.block_size, self.block_size, self.block_size)
                    pygame.draw.rect(screen, colors[row][col], rect)
                    pygame.draw.rect(screen, (0, 0, 0), rect, 1)

                    if grid[row][col] == 'W':
                        continue
                    message = font.render(array[row][col], True, (0, 0, 0))
                    screen.blit(message, (col * self.block_size + offset[0] * DisplayManager.RATIO, row * self.block_size + offset[1] * DisplayManager.RATIO))

            pygame.display.update()

            if save:
                pygame.image.save(screen, file_name)

