import pygame, sys
from pygame.locals import *
from random import randint
from neat import Population, NEATConfig
from numpy import argmax
import multiprocessing as mp
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import matplotlib.pyplot as plt

WIDTH, HEIGHT = 600, 600
CELL_SIZE = 50
FPS = 10

COLORS = {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
}

class Snake:
    def __init__(self, cell_height, cell_width, color, surface=None):
        self.body = []
        self.surface = surface
        self.cell_height = cell_height
        self.cell_width = cell_width
        self.color = color
        self.direction = 1  # 0=Up, 1=Right, 2=Down, 3=Left

    @property
    def grid_height(self): return HEIGHT // self.cell_height
    @property
    def grid_width(self): return WIDTH // self.cell_width

    def draw(self):
        head_r, head_c = self.body[-1]
        pygame.draw.rect(self.surface, COLORS['blue'],
                         (head_c*self.cell_width, head_r*self.cell_height,
                          self.cell_width, self.cell_height))
        for seg in self.body[:-1]:
            r, c = seg
            pygame.draw.rect(self.surface, self.color,
                             (c*self.cell_width, r*self.cell_height,
                              self.cell_width, self.cell_height))

    def move(self, action):
        # action: 0 = left turn, 1 = straight, 2 = right turn
        self.direction = (self.direction + (action - 1)) % 4
        dr, dc = [(-1,0),(0,1),(1,0),(0,-1)][self.direction]
        head_r, head_c = self.body[-1]
        new_head = (head_r+dr, head_c+dc)
        self.body.append(new_head)
        self.body.pop(0)

    def add_tail(self):
        tail = self.body[0]
        self.body.insert(0, tail)

class SnakeGame:
    def __init__(self, cell_height, cell_width, surface=None, show_grid=False, snake_color=(0,255,0), use_ray_cast=False):
        self.surface = surface
        self.cell_height = cell_height
        self.cell_width = cell_width
        self.show_grid = show_grid
        self.snake_color = snake_color
        self.use_ray_cast = use_ray_cast
        self.reset()

    @property
    def grid_height(self):
        return HEIGHT // self.cell_height

    @property
    def grid_width(self):
        return WIDTH // self.cell_width

    def reset(self):
        self.snake = Snake(self.cell_height, self.cell_width, self.snake_color, self.surface)
        mid = (self.grid_height//2, self.grid_width//2)
        self.snake.body = [(mid[0], mid[1]-1), mid]
        self.apple = self._spawn_apple()
        self.game_over = False
        self.steps = 0
        self.apple_eaten = 0
        self.steps_since_last_apple = 0
        self.max_without_apple = 100

    def _spawn_apple(self):
        while True:
            pos = (randint(0,self.grid_height-1), randint(0,self.grid_width-1))
            if pos not in self.snake.body:
                return pos

    def ray_cast(self, start_r, start_c, dr, dc):
        max_dist = max(self.grid_height, self.grid_width)
        found_apple = found_body = found_wall = 0.0
        dist = 0.0
        for i in range(1, max_dist+1):
            r, c = start_r + dr*i, start_c + dc*i
            dist = float(i)/max_dist
            if not (0 <= r < self.grid_height and 0 <= c < self.grid_width):
                found_wall = 1.0
                break
            if (r,c) == self.apple:
                found_apple = 1.0
                break
            if (r,c) in self.snake.body:
                found_body = 1.0
                break
        return [found_apple, found_body, found_wall, dist]

    def get_vision(self):
        head_r, head_c = self.snake.body[-1]
        rays = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
        vision = []
        for dr, dc in rays:
            vision.extend(self.ray_cast(head_r, head_c, dr, dc))
        return vision

    def get_state(self):
        head_r, head_c = self.snake.body[-1]
        apple_r, apple_c = self.apple
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        forward = dirs[self.snake.direction]
        left = dirs[(self.snake.direction - 1) % 4]
        right = dirs[(self.snake.direction + 1) % 4]

        def danger(offset):
            r, c = head_r + offset[0], head_c + offset[1]
            return int(r < 0 or r >= self.grid_height or c < 0 or c >= self.grid_width or (r, c) in self.snake.body)

        return [
            danger(forward),
            danger(left),
            danger(right),
            int(apple_r < head_r), int(apple_r > head_r),
            int(apple_c < head_c), int(apple_c > head_c),
            int(self.snake.direction == 0), int(self.snake.direction == 1),
            int(self.snake.direction == 2), int(self.snake.direction == 3)
        ]

    def update(self):
        head = self.snake.body[-1]
        if head in self.snake.body[:-1] or not (0<=head[0]<self.grid_height and 0<=head[1]<self.grid_width):
            self.game_over = True
            return False
        if head == self.apple:
            self.snake.add_tail()
            self.apple = self._spawn_apple()
            self.apple_eaten += 1
            self.steps_since_last_apple = 0
            return True
        self.steps += 1
        self.steps_since_last_apple += 1
        if self.steps_since_last_apple > self.max_without_apple:
            self.game_over = True
        return False

    def draw(self):
        if not self.surface: return
        self.surface.fill(COLORS['black'])
        if self.show_grid:
            for i in range(self.grid_width+1):
                pygame.draw.line(self.surface, COLORS['white'], (i*self.cell_width,0), (i*self.cell_width,HEIGHT))
            for j in range(self.grid_height+1):
                pygame.draw.line(self.surface, COLORS['white'], (0,j*self.cell_height), (WIDTH,j*self.cell_height))
        apple_r, apple_c = self.apple
        pygame.draw.rect(self.surface, COLORS['red'], (apple_c*self.cell_width, apple_r*self.cell_height, self.cell_width, self.cell_height))
        self.snake.draw()
        pygame.display.flip()

    def get_fitness(self):
        return self.apple_eaten*100 + self.steps - self.steps_since_last_apple

def neat_play(genome):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('NEAT Snake Viz')
    clock = pygame.time.Clock()
    game = SnakeGame(CELL_SIZE, CELL_SIZE, surface=screen, show_grid=True)
    while not game.game_over:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("Escape key pressed. Exiting visualization...")
                    game.game_over = True
        state = game.get_vision() if game.use_ray_cast else game.get_state()
        output = genome.forward(state)
        action = argmax(output)
        game.snake.move(action)
        game.update()
        game.draw()
        clock.tick(FPS)
    pygame.quit()
    print(f"Visualization finished. Score: {game.apple_eaten}, Steps: {game.steps}")

def eval_genome(genome):
    game = SnakeGame(CELL_SIZE, CELL_SIZE)
    while not game.game_over and game.steps < 1000:
        state = game.get_vision() if game.use_ray_cast else game.get_state()
        output = genome.forward(state)
        action = argmax(output)
        game.snake.move(action)
        game.update()
    return game.get_fitness()

def training_loop(generations=100, pop_size=200, viz_step=10, num_processes=8):
    input_size = len(SnakeGame(CELL_SIZE, CELL_SIZE, use_ray_cast=False).get_state())
    output_size = 3
    config = NEATConfig(
        population_size=pop_size,
        genome_shape=(input_size, output_size),
        hid_node_activation='relu',
        out_node_activation='sigmoid'
    )
    print(config)
    pop = Population(config=config)
    avg_fitness, top_fitness = [], []

    for gen in range(generations):
        print(f"--- Generation {gen+1}/{generations} ---")
        with mp.Pool(num_processes) as pool:
            results = pool.map(eval_genome, pop.members)
        for genome, fit in zip(pop.members, results): genome.fitness = fit
        avg = sum(results)/len(results)
        top = max(results)
        avg_fitness.append(avg); top_fitness.append(top)
        print(f"Avg: {avg:.2f}, Top: {top:.2f}")
        if (gen+1)%viz_step==0 or gen+1==generations:
            print(f"Visualizing gen {gen+1}")
            top_genome = pop.get_top_genome()
            neat_play(top_genome)
            plt.figure()
            plt.plot(range(1, len(avg_fitness)+1), avg_fitness)
            plt.plot(range(1, len(top_fitness)+1), top_fitness)
            plt.legend(['Avg','Top']); plt.xlabel('Gen'); plt.ylabel('Fitness'); plt.show()
        pop.reproduce()

    top_genome = pop.get_top_genome()
    print(f"Finished. Best fitness: {top_genome.fitness}")
    return top_genome

if __name__ == '__main__':
    best = training_loop(generations=100, pop_size=500, viz_step=10, num_processes=8)
    neat_play(best)
    pygame.quit()
    sys.exit()
