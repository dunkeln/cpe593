import pygame
from lib import maze

def main():
    pygame.init()
    X, Y = 720, 720
    screen = pygame.display.set_mode((X, Y))
    clock = pygame.time.Clock()
    running = True

    obstacles = list(maze.randomize_obstacles(*screen.get_size(), 100))
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
        # INFO: we at 20 FPS
        clock.tick(20)

    pygame.quit()

if __name__ == "__main__":
    main()
