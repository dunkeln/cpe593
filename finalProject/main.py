import pygame
from lib import maze

def main():
    pygame.init()
    X, Y = 720, 720
    screen = pygame.display.set_mode((X, Y))
    clock = pygame.time.Clock()
    running = True

    # INFO: global settings
    FPS = 20
    GRID_RES = 20

    obstacles = list(maze.randomize_obstacles(*screen.get_size(), 100, grid_size=GRID_RES))
    print(f"generated {len(obstacles)} obstacles")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill("black")

        # INFO: RENDER YOUR GAME HERE
        maze.draw(screen, obstacles, color="white")

        # flip() the display to put your work on screen
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
