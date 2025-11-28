"""
PCB Routing CSP Solver v2.0 (Smart & Laconic)
---------------------------------------------
Покращення:
1. Мінімізація поворотів (шляхи стають прямими).
2. Спрямований пошук (шляхи стають коротшими).

"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Tuple, Dict, Set
import random

Point = Tuple[int, int]
Path = List[Point]

class PCBGrid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.obstacles: Set[Point] = set()
        self.terminals: Dict[int, Tuple[Point, Point]] = {}
        self.colors: Dict[int, str] = {}

    def add_obstacle(self, x: int, y: int):
        self.obstacles.add((x, y))

    def add_net(self, net_id: int, start: Point, end: Point, color: str):
        self.terminals[net_id] = (start, end)
        self.colors[net_id] = color

    def is_valid(self, p: Point) -> bool:
        x, y = p
        return 0 <= x < self.width and 0 <= y < self.height and p not in self.obstacles

# === НОВА ЛОГІКА: ОЦІНКА "КРАСИ" ШЛЯХУ ===

def count_turns(path: Path) -> int:
    """Рахує кількість поворотів у шляху. Менше = краще."""
    if len(path) < 3:
        return 0
    turns = 0
    # Перевіряємо трійки точок. Якщо вони не на одній лінії - це поворот.
    for i in range(len(path) - 2):
        p1 = path[i]
        p2 = path[i+1]
        p3 = path[i+2]
        
        # Вектори руху
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        if v1 != v2:
            turns += 1
    return turns

class PathGenerator:
    """Розумний генератор маршрутів"""
    
    @staticmethod
    def generate_candidate_paths(grid: PCBGrid, net_id: int, max_paths: int = 50) -> List[Path]:
        start, end = grid.terminals[net_id]
        manhattan_dist = abs(start[0] - end[0]) + abs(start[1] - end[1])
        max_len = manhattan_dist + 10  # Невеликий запас, щоб не блукав
        
        paths = []
        stack = [(start, [start])]
        
        attempts = 0
        MAX_ATTEMPTS = 1000
        
        while stack and len(paths) < max_paths and attempts < MAX_ATTEMPTS:
            attempts += 1
            curr, path = stack.pop() # DFS
            
            if len(path) > max_len:
                continue
                
            if curr == end:
                paths.append(path)
                continue
            
            # === ПОКРАЩЕННЯ 1: Спрямований рух ===
            # Генеруємо сусідів
            neighbors = []
            moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(moves) # Все ще потрібен рандом для варіативності
            
            for dx, dy in moves:
                nx, ny = curr[0] + dx, curr[1] + dy
                next_p = (nx, ny)
                
                if grid.is_valid(next_p) and next_p not in path:
                    # Перевірка на чужі піни
                    is_blocked = False
                    for other_id, (os, oe) in grid.terminals.items():
                        if other_id != net_id and (next_p == os or next_p == oe):
                            is_blocked = True
                            break
                    if not is_blocked:
                        # Рахуємо відстань до цілі для цього сусіда
                        dist_to_end = abs(nx - end[0]) + abs(ny - end[1])
                        neighbors.append((dist_to_end, next_p))
            
            # Сортуємо сусідів: спочатку ті, що ближче до цілі (Greedy bias)
            # Але іноді (20% випадків) перемішуємо, щоб знайти обхідні шляхи
            neighbors.sort(key=lambda x: x[0], reverse=True) # Reverse, бо pop бере з кінця
            
            if random.random() < 0.2: 
                random.shuffle(neighbors)

            for _, next_p in neighbors:
                stack.append((next_p, path + [next_p]))
                        
        # === ПОКРАЩЕННЯ 2: Сортування за "Лаконічністю" ===
        # Критерій: Довжина + (Кількість поворотів * 2)
        # Це змушує алгоритм обирати прямі лінії, навіть якщо вони трохи довші
        paths.sort(key=lambda p: len(p) + count_turns(p) * 2)
        
        # Обрізаємо домен, залишаючи тільки найкращі (найпряміші) варіанти
        return paths[:50] 

class PCBSolver:
    def __init__(self, grid: PCBGrid):
        self.grid = grid
        self.domains: Dict[int, List[Path]] = {} 
        self.assignment: Dict[int, Path] = {}
        self.stats_nodes = 0

    def prepare_domains(self):
        for net_id in self.grid.terminals:
            paths = PathGenerator.generate_candidate_paths(self.grid, net_id)
            if not paths: return False
            self.domains[net_id] = paths
        return True

    def solve(self) -> bool:
        if not self.prepare_domains(): return False
        return self._backtrack()

    def _backtrack(self) -> bool:
        self.stats_nodes += 1
        if len(self.assignment) == len(self.grid.terminals): return True

        # MRV Heuristic
        unassigned = [nid for nid in self.grid.terminals if nid not in self.assignment]
        var = min(unassigned, key=lambda nid: len(self.domains[nid]))
        
        for path in self.domains[var]:
            if self._is_consistent(path):
                self.assignment[var] = path
                if self._backtrack(): return True
                del self.assignment[var]
        return False

    def _is_consistent(self, new_path: Path) -> bool:
        new_path_set = set(new_path)
        for assigned_path in self.assignment.values():
            if not new_path_set.isdisjoint(assigned_path): return False
        return True

# === VIZUALIZATION (Canvas Style) ===

class PCBVisualizer:
    @staticmethod
    def draw(grid: PCBGrid, solution: Dict[int, Path] = None):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Темний фон "Blueprint"
        ax.set_facecolor('#1e1e24') 
        
        # Тонка сітка
        ax.set_xlim(-0.5, grid.width - 0.5)
        ax.set_ylim(-0.5, grid.height - 0.5)
        ax.set_xticks(np.arange(grid.width))
        ax.set_yticks(np.arange(grid.height))
        ax.grid(True, color='#333333', linestyle='-', linewidth=0.5)
        
        # Перешкоди (штриховка)
        for ox, oy in grid.obstacles:
            rect = mpatches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, 
                                    facecolor='#2a2a2a', edgecolor='#444444', hatch='///')
            ax.add_patch(rect)
            
        # Дроти
        for net_id, (start, end) in grid.terminals.items():
            color = grid.colors[net_id]
            
            # Піни (Contacts) - з білим обідком
            for p in [start, end]:
                ax.add_patch(plt.Circle(p, 0.3, color=color, ec='white', lw=2, zorder=10))
            
            # Підписи
            ax.text(start[0], start[1], str(net_id), color='white', ha='center', va='center', fontweight='bold', zorder=11, fontsize=9)
            ax.text(end[0], end[1], str(net_id), color='white', ha='center', va='center', fontweight='bold', zorder=11, fontsize=9)
            
            if solution and net_id in solution:
                path = solution[net_id]
                xs = [p[0] for p in path]
                ys = [p[1] for p in path]
                
                # Основна лінія (більш "лаконічна", без прозорості)
                ax.plot(xs, ys, color=color, linewidth=4, alpha=1.0, zorder=5, 
                        solid_capstyle='round', solid_joinstyle='round')
                
                # Тонка біла лінія всередині (ефект дроту)
                ax.plot(xs, ys, color='white', linewidth=1, alpha=0.3, zorder=6)

        # Прибираємо осі
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        plt.tight_layout()
        plt.show()

# === GENERATOR ===

def generate_random_layout(width: int, height: int, n_obstacles: int, n_nets: int) -> PCBGrid:
    """Генерує складну розкладку з перешкодами в ЦЕНТРІ"""
    grid = PCBGrid(width, height)
    used_positions = set()

    # 1. Генерація перешкод (Тепер суворо всередині поля!)
    # Ми беремо діапазон від 1 до width-2, щоб не ставити перешкоди на самій рамці
    while len(grid.obstacles) < n_obstacles:
        rx = random.randint(1, width - 2)
        ry = random.randint(1, height - 2)
        
        if (rx, ry) not in used_positions:
            grid.add_obstacle(rx, ry)
            used_positions.add((rx, ry))

    # 2. Генерація мереж
    colors = ['#ff4757', '#2ed573', '#1e90ff', '#ffa502', '#a55eea', '#ffffff']
    
    for i in range(n_nets):
        max_tries = 1000
        for _ in range(max_tries):
            # Старт і фініш теж намагаємось розкидати по всьому полю
            sx = random.randint(0, width - 1)
            sy = random.randint(0, height - 1)
            
            if (sx, sy) in used_positions: continue

            # Шукаємо кінець на відстані
            dist = random.randint(4, width) # Мінімальна довжина дроту
            angle = random.choice([0, 90, 180, 270]) # Тільки прямі кути для генерації
            # Додаємо трохи випадкового зміщення
            ex = sx + random.randint(-dist, dist)
            ey = sy + random.randint(-dist, dist)
            
            # Перевіряємо, чи точка валідна
            if (0 <= ex < width and 0 <= ey < height and 
                (ex, ey) not in used_positions and 
                abs(sx-ex) + abs(sy-ey) > 4): # Дріт має бути довгим
                
                used_positions.add((sx, sy))
                used_positions.add((ex, ey))
                
                color = colors[i % len(colors)]
                grid.add_net(i + 1, (sx, sy), (ex, ey), color)
                break
                
    return grid

def main():
    # Налаштування для "Цікавої" гри
    BOARD_SIZE = 12
    N_OBSTACLES = 8       # Більше перешкод (було 4)
    N_NETS = 5            # 5 дротів
    
    print("Шукаємо рішення з перешкодами по центру...")
    
    # Даємо більше спроб, бо тепер згенерувати валідну плату важче
    for i in range(200):
        grid = generate_random_layout(BOARD_SIZE, BOARD_SIZE, N_OBSTACLES, N_NETS)
        solver = PCBSolver(grid)
        
        # Використовуємо solver
        if solver.solve():
            print(f"Знайдено! Спроба #{i+1}")
            print(f"Статистика: {solver.stats_nodes} вузлів")
            PCBVisualizer.draw(grid, solver.assignment)
            break
    else:
        print("Не знайдено складного рішення. Спробуйте ще раз.")

if __name__ == "__main__":
    main()
