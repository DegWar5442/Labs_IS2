import pygame
import sys
import random
import numpy as np
from maze_generator import generate_maze
from pathfinding import maze_to_graph, a_star_search, dijkstra_search, bfs_search
from agents import PacMan, Ghost
from heuristics import heuristic_manhattan, heuristic_euclidean
from utils import load_image
import time

# Initialize Pygame
pygame.init()

# Constants
CELL_SIZE = 20
FPS = 12 # Збільшено FPS для кращої анімації
INITIAL_WIDTH = 30
INITIAL_HEIGHT = 30
MAX_LEVEL = 10
POINTS_TO_ADVANCE = 300
INITIAL_GHOST_SPEED = 0.25
GHOST_SPEED_INCREASE_RATE = (1.0 - INITIAL_GHOST_SPEED) / 4

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
PACMAN_COLOR = (255, 255, 0)
GHOST_COLOR = (255, 0, 0)
PELLET_COLOR = (255, 255, 255)

# Load images
# PACMAN_IMG = load_image('assets/pacman.png', (CELL_SIZE, CELL_SIZE)) # Не використовується
# GHOST_IMG = load_image('assets/ghost.png', (CELL_SIZE, CELL_SIZE)) # Не використовується
WALL_IMG = load_image('assets/wall.png', (CELL_SIZE, CELL_SIZE))
DEATH_IMG = load_image('assets/pacmanDeath.png', (CELL_SIZE, CELL_SIZE))
ARROW_IMG = load_image('assets/arrow.png', (CELL_SIZE *2  , CELL_SIZE*3))

# --- Завантаження спрайтів Pac-Man (Рух) ---
PACMAN_ANIM_FRAMES = []
for i in range(1, 4): # pacman01.png, pacman02.png, pacman03.png
    try:
        PACMAN_ANIM_FRAMES.append(load_image(f'assets/pacman0{i}.png', (CELL_SIZE, CELL_SIZE)))
    except Exception as e:
        print(f"Помилка завантаження pacman0{i}.png: {e}")

# --- Завантаження спрайтів Привидів (Кольори) ---
GHOST_IMAGES = []
for i in range(1, 5): # ghost1.png, ghost2.png, ghost3.png, ghost4.png
    try:
        GHOST_IMAGES.append(load_image(f'assets/ghost0{i}.png', (CELL_SIZE, CELL_SIZE)))
    except Exception as e:
        print(f"Помилка завантаження ghost{i}.png: {e}")
# ----------------------------------------------------

class Game:
    def __init__(self):
        self.is_countdown = False
        self.countdown_start_time = 0
        self.reset_game()
        self.game_time = 0
        
    def set_auto_mode(self, vl):
        self.auto_mode = vl

    def reset_game(self):
        self.level = 1
        self.width = INITIAL_WIDTH
        self.height = INITIAL_HEIGHT
        self.screen_width = self.width * CELL_SIZE
        self.screen_height = self.height * CELL_SIZE + 60
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pac-Man")
        self.clock = pygame.time.Clock()
        self.maze = generate_maze(self.width, self.height)
        self.graph = maze_to_graph(self.maze)
        self.pacman = self.initialize_pacman()
        self.ghosts = self.initialize_ghosts()
        self.running = False
        self.game_over = False
        self.score = 0
        self.level_score = 0
        self.pellets = self.initialize_pellets()
        self.font = pygame.font.Font(None, 36)
        self.ghost_speed = INITIAL_GHOST_SPEED
        self.ghost_move_counter = 0
        self.auto_mode = False
        self.game_time = 0
        
        # Починаємо гру з відліку
        self.is_countdown = True
        self.countdown_start_time = time.time()
        
    def toggle_auto_mode(self):
        self.auto_mode = not self.auto_mode
        mode_text = "Automatic" if self.auto_mode else "Manual"
        print(f"Switched to {mode_text} mode")

    def initialize_pacman(self):
        open_cells = list(zip(*np.where(self.maze == 1)))
        start_position = random.choice(open_cells)
        # Передаємо кадри анімації та смерті
        return PacMan(start_position, self.graph, 
                      anim_frames=PACMAN_ANIM_FRAMES, 
                      death_image=DEATH_IMG)

    def initialize_ghosts(self):
        ghosts = []
        desired_num_ghosts = min(4, self.level + 1)
        open_cells = list(zip(*np.where(self.maze == 1)))
        if self.pacman.position in open_cells:
            open_cells.remove(self.pacman.position)
        available_cells = len(open_cells)
        num_ghosts = min(available_cells, desired_num_ghosts)
        if num_ghosts > 0:
            ghost_positions = random.sample(open_cells, num_ghosts)
            for i, pos in enumerate(ghost_positions):
                strategy = 'chase' if random.random() > 0.5 else 'ambush'
                ghosts.append(Ghost(pos, self.graph, strategy, ghost_id=i)) # Передаємо ID
        return ghosts

    def initialize_pellets(self):
        pellets = set()
        open_cells = list(zip(*np.where(self.maze == 1)))
        num_pellets = len(open_cells) // 2
        pellet_positions = random.sample(open_cells, num_pellets)
        for pos in pellet_positions:
            pellets.add(pos)
        return pellets

    def increase_level(self):
        self.level += 1
        self.width += 2
        self.height += 2
        self.screen_width = self.width * CELL_SIZE
        self.screen_height = self.height * CELL_SIZE + 60
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.maze = generate_maze(self.width, self.height)
        self.graph = maze_to_graph(self.maze)
        self.pacman = self.initialize_pacman()
        self.ghosts = self.initialize_ghosts()
        self.pellets = self.initialize_pellets()
        self.level_score = 0
        self.show_intermediate_window()
        # --- НОВЕ: Запуск зворотного відліку та стрілки ---
        self.is_countdown = True
        self.countdown_start_time = time.time()
        # ---------------------------------------------------
        self.update_ghost_speed()
        

    def update_ghost_speed(self):
        if self.level < 5:
            self.ghost_speed = INITIAL_GHOST_SPEED + GHOST_SPEED_INCREASE_RATE * (self.level - 1)
        else:
            self.ghost_speed = 1.0

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    self.set_auto_mode(True)
                elif event.key == pygame.K_j:
                    self.set_auto_mode(False)

        if not self.auto_mode and not self.pacman.is_dead:
            keys = pygame.key.get_pressed()
            direction = None
            if keys[pygame.K_LEFT]: direction = (0, -1)
            elif keys[pygame.K_RIGHT]: direction = (0, 1)
            elif keys[pygame.K_UP]: direction = (-1, 0)
            elif keys[pygame.K_DOWN]: direction = (1, 0)
            
            if direction:
                new_position = (self.pacman.position[0] + direction[0],
                                self.pacman.position[1] + direction[1])
                if self.is_walkable(new_position):
                    self.pacman.move(new_position)
        elif self.auto_mode and not self.pacman.is_dead:
            new_position = self.pacman.decide_move(self.graph, self.pellets, self.ghosts)
            if self.is_walkable(new_position):
                self.pacman.move(new_position)

        if self.pacman.position in self.pellets:
            self.pellets.remove(self.pacman.position)
            self.score += 10
            self.level_score += 10
            if self.level_score >= POINTS_TO_ADVANCE:
                self.increase_level()
                
    def is_walkable(self, position):
        y, x = position
        if 0 <= y < self.height and 0 <= x < self.width:
            return self.maze[y, x] == 1
        return False

    def update_ghosts(self):
        if self.pacman.is_dead and self.pacman.death_timer <= 0:
            return
            
        self.ghost_move_counter += self.ghost_speed
        if self.ghost_move_counter >= 1:
            self.ghost_move_counter -= 1
            for ghost in self.ghosts:
                new_position = ghost.decide_move(self.pacman.position, self.game_time)
                if self.is_walkable(new_position):
                    ghost.move(new_position)
        self.game_time += 1

    def update_countdown(self):
        current_time = time.time()
        elapsed = current_time - self.countdown_start_time
        
        if elapsed >= 2:
            self.is_countdown = False

    def check_collisions(self):
        if self.pacman.is_dead:
            return

        for ghost in self.ghosts:
            if ghost.position == self.pacman.position:
                self.pacman.die() # Pac-Man переходить у стан смерті
                
    def handle_death(self):
        if self.pacman.is_dead:
            self.pacman.death_timer -= 1
            if self.pacman.death_timer <= 0:
                self.pacman.restore() # Скидаємо позицію Pac-Man
                # 1. Скидання прогресу рівня
                self.level_score = 0 
            
            # 2. Відновлення кульок
                self.pellets = self.initialize_pellets() # Перестворює кульки для поточного лабіринту
            # --------------------
                # Скидаємо позиції привидів
                open_cells = list(zip(*np.where(self.maze == 1)))
                for ghost in self.ghosts:
                    ghost.position = random.choice(open_cells)
                    if ghost.position in open_cells: open_cells.remove(ghost.position)
                
                # Починаємо відлік знову
                self.is_countdown = True
                self.countdown_start_time = time.time()
                
    def render(self):
        self.screen.fill(BLACK)
        for y in range(self.height):
            for x in range(self.width):
                if self.maze[y, x] == 0:
                    if WALL_IMG:
                        self.screen.blit(WALL_IMG, (x*CELL_SIZE, y*CELL_SIZE))
                    else:
                        pygame.draw.rect(self.screen, BLACK, pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
                else:
                    pygame.draw.rect(self.screen, BLACK, pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        for pellet in self.pellets:
            pygame.draw.circle(self.screen, PELLET_COLOR, 
                               (pellet[1]*CELL_SIZE + CELL_SIZE//2, 
                                pellet[0]*CELL_SIZE + CELL_SIZE//2), 
                               CELL_SIZE//4)
        
        # --- РЕНДЕРИНГ PAC-MAN (Анімація та Смерть) ---
        current_pacman_image = None
        if self.pacman.is_dead and self.pacman.death_timer > 0:
            current_pacman_image = DEATH_IMG
        elif PACMAN_ANIM_FRAMES:
            current_pacman_image = PACMAN_ANIM_FRAMES[self.pacman.current_frame]
        if current_pacman_image:
            # Зберігаємо координати Pac-Man, які ми використовуємо для стрілки
            pacman_draw_x = self.pacman.position[1]*CELL_SIZE
            pacman_draw_y = self.pacman.position[0]*CELL_SIZE
            
            self.screen.blit(current_pacman_image, (pacman_draw_x, pacman_draw_y))
        if current_pacman_image:
            self.screen.blit(current_pacman_image, 
                             (self.pacman.position[1]*CELL_SIZE, self.pacman.position[0]*CELL_SIZE))
        # -----------------------------------------------

        # --- РЕНДЕРИНГ ПРИВИДІВ (За кольором) ---
        for ghost in self.ghosts:
            if GHOST_IMAGES and ghost.ghost_id < len(GHOST_IMAGES):
                current_ghost_image = GHOST_IMAGES[ghost.ghost_id]
                self.screen.blit(current_ghost_image, (ghost.position[1]*CELL_SIZE, ghost.position[0]*CELL_SIZE))
            else:
                # Резервне малювання
                pygame.draw.circle(self.screen, GHOST_COLOR, 
                                   (ghost.position[1]*CELL_SIZE + CELL_SIZE//2, 
                                    ghost.position[0]*CELL_SIZE + CELL_SIZE//2), 
                                   CELL_SIZE//2)
        # -----------------------------------------
        
        # --- РЕНДЕРИНГ ВІДЛІКУ ---
        if self.is_countdown:
            # 1. МАЛЮВАННЯ СТРІЛКИ ПІД ЧАС ВІДЛІКУ (НОВЕ)
            if ARROW_IMG and current_pacman_image:
                # Малюємо стрілку прямо над позицією Pac-Man
                arrow_x = pacman_draw_x + (CELL_SIZE - ARROW_IMG.get_width()) // 2
                arrow_y = pacman_draw_y - ARROW_IMG.get_height() + 5

                self.screen.blit(ARROW_IMG, (arrow_x, arrow_y))
            current_count = 2 - int(time.time() - self.countdown_start_time)
            
            if current_count < 1:
                countdown_text = self.font.render("GO!", True, WHITE)
            else:
                countdown_text = self.font.render(str(current_count), True, WHITE)

            text_rect = countdown_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            self.screen.blit(countdown_text, text_rect)
        # -------------------------

        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        level_text = self.font.render(f"Level: {self.level}", True, WHITE)
        level_progress_text = self.font.render(f"Level Progress: {self.level_score}/{POINTS_TO_ADVANCE}", True, WHITE)
        ghost_speed_text = self.font.render(f"Ghost Speed: {self.ghost_speed:.2f}", True, WHITE)
        mode_text = self.font.render(f"Mode: {'Auto' if self.auto_mode else 'Manual'}", True, WHITE)
        controls_text = self.font.render("H: Auto  J: Manual", True, WHITE)

        self.screen.blit(score_text, (10, self.height * CELL_SIZE + 5))
        self.screen.blit(level_text, (self.width * CELL_SIZE - 100, self.height * CELL_SIZE + 5))
        self.screen.blit(level_progress_text, (10, self.height * CELL_SIZE + 25))
        self.screen.blit(ghost_speed_text, (self.width * CELL_SIZE - 200, self.height * CELL_SIZE + 25))
        self.screen.blit(mode_text, (10, self.height * CELL_SIZE + 45))
        self.screen.blit(controls_text, (self.width * CELL_SIZE - 300, self.height * CELL_SIZE + 45))
        
        pygame.display.flip()

    def show_menu(self, title, options):
        menu_font = pygame.font.Font(None, 48)
        selected_option = 0
        
        while True:
            self.screen.fill(BLACK)
            title_surface = menu_font.render(title, True, WHITE)
            title_rect = title_surface.get_rect(center=(self.screen_width // 2, 100))
            self.screen.blit(title_surface, title_rect)
            
            for i, option in enumerate(options):
                color = PACMAN_COLOR if i == selected_option else WHITE
                option_surface = self.font.render(option, True, color)
                option_rect = option_surface.get_rect(center=(self.screen_width // 2, 200 + i * 50))
                self.screen.blit(option_surface, option_rect)
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selected_option = (selected_option - 1) % len(options)
                    elif event.key == pygame.K_DOWN:
                        selected_option = (selected_option + 1) % len(options)
                    elif event.key == pygame.K_RETURN:
                        return options[selected_option]

    def show_intermediate_window(self):
        self.screen.fill(BLACK)
        title_font = pygame.font.Font(None, 64)
        info_font = pygame.font.Font(None, 36)

        title_text = title_font.render(f"Level {self.level} Complete!", True, WHITE)
        score_text = info_font.render(f"Total Score: {self.score}", True, WHITE)
        next_level_text = info_font.render(f"Preparing for Level {self.level + 1}...", True, WHITE)
        continue_text = info_font.render("Press SPACE to continue", True, WHITE)

        self.screen.blit(title_text, (self.screen_width // 2 - title_text.get_width() // 2, 100))
        self.screen.blit(score_text, (self.screen_width // 2 - score_text.get_width() // 2, 200))
        self.screen.blit(next_level_text, (self.screen_width // 2 - next_level_text.get_width() // 2, 250))
        self.screen.blit(continue_text, (self.screen_width // 2 - continue_text.get_width() // 2, 350))

        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    waiting = False

    def run(self):
        while True:
            action = self.show_menu("Pac-Man", ["Start", "Quit"])
            if action == "Start":
                self.reset_game()
                self.game_loop()
            elif action == "Quit":
                break
        
        pygame.quit()
        sys.exit()

    def game_loop(self):
        self.running = True
        self.game_over = False
        while self.running and self.level <= MAX_LEVEL:
            
            if self.is_countdown:
                self.update_countdown()
                self.render()
                self.clock.tick(FPS)
                continue
                
            self.handle_input()
            self.pacman.update_animation()
            self.handle_death()
            
            self.update_ghosts()
            self.check_collisions()
            self.render()
            
            self.clock.tick(FPS)
        
        if self.game_over:
            self.show_menu("Game Over", ["Restart", "Quit"])
        elif self.level > MAX_LEVEL:
            self.show_menu("!!YAY!!", ["Restart", "Quit"])
            
if __name__ == "__main__":
    game = Game()
    game.run()