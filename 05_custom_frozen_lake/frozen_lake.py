import pygame
import random

WIDTH, HEIGHT = 400 , 400
ROW, COLUMN = 10, 10
FPS = 10
BOX_WIDTH, BOX_HEIGHT = 40, 40

score = 0
heroPosition = [5, 4]
goalPosition = [3, 3]
enemyPosition = [2, 2]
gameOver = False

def collision(heroPosition, goalPosition, enemyPosition, score, gameOver):
  
    if heroPosition == goalPosition:
        score += 1
        goalPosition = [random.randrange(1, COLUMN), random.randrange(1, ROW)]
    
    gameOver = False
    if heroPosition == enemyPosition:
        gameOver = True

    # Collision with top wall
    if heroPosition[1] < 0:
        gameOver = True

    # Collision with bottom wall
    if heroPosition[1] > ROW - 1:
        gameOver = True
    
    # Collision with left wall
    if heroPosition[0] < 0:
        gameOver = True
    
    # Collision with right wall
    if heroPosition[0] > COLUMN - 1:
        gameOver = True

    return goalPosition, score, gameOver


pygame.init()
pygame.display.set_caption('Frozen lake 10 x 10')
display = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial_bold', 200)
running = True


while running:

    goalPosition, score, gameOver = collision(heroPosition, goalPosition, enemyPosition, score, gameOver)
    
    # Movement
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                break
            if event.key == pygame.K_RIGHT:
                heroPositionX = heroPosition[0] + 1
                heroPositionY = heroPosition[1]
            if event.key == pygame.K_LEFT:
                heroPositionX = heroPosition[0] - 1
                heroPositionY = heroPosition[1]
            if event.key == pygame.K_UP:
                heroPositionX = heroPosition[0]
                heroPositionY = heroPosition[1] - 1
            if event.key == pygame.K_DOWN:
                heroPositionX = heroPosition[0]
                heroPositionY = heroPosition[1] + 1
            heroPosition = [heroPositionX, heroPositionY]

    # Draw background
    display.fill((67, 70, 75))

    # Draw score or game over
    if gameOver:
        img = font.render('Game Over', True, (57, 60, 65))
    else:
        img = font.render(str(score), True, (57, 60, 65))

    display.blit(img, (WIDTH // 2 - 100, HEIGHT // 2 - 100))

    # Draw Hero
    pygame.draw.rect(display, 'BLUE', (heroPosition[0] * 40, heroPosition[1] * 40, BOX_WIDTH, BOX_HEIGHT))

    # Draw Goal
    pygame.draw.rect(display, 'GREEN', (goalPosition[0] * 40, goalPosition[1] * 40, BOX_WIDTH, BOX_HEIGHT))

    # Draw Enemy
    pygame.draw.rect(display, 'RED', (enemyPosition[0] * 40, enemyPosition[1] * 40, BOX_WIDTH, BOX_HEIGHT))

    pygame.display.update()
    clock.tick(FPS)

pygame.quit()