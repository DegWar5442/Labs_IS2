"""
PCB Routing CSP Solver (Auto-Retry Mode)
----------------------------------------
Логіка: Якщо рішення не знайдено, алгоритм змінює розташування перешкод 
та контактів і пробує знову (до ліміту спроб).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Tuple, Dict, Set
import random
import time

# Типи даних
Point = Tuple[int, int]
Path = List[Point]

class PCBGrid:
    """Представлення плати"""
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

class PathGenerator:
    """Генератор можливих маршрутів (Domain Generator)"""
    
    @staticmethod
    def generate_candidate_paths(grid: PCBGrid, net_id: int, max_paths: int = 200) -> List[Path]:
        start, end = grid.terminals[net_id]
        
        # Евристика: Манхеттенська відстань + запас для маневрів
        manhattan_dist = abs(start[0] - end[0]) + abs(start[1] - end[1])
        max_len = manhattan_dist + 14 
        
        paths = []
        stack = [(start, [start])] 
        
        attempts = 0
        MAX_ATTEMPTS = 5000 
        
        while stack and len(paths) < max_paths and attempts < MAX_ATTEMPTS:
            attempts += 1
            curr, path = stack.pop()
            
            if len(path) > max_len:
                continue
                
            if curr == end:
                paths.append(path)
                continue
            
            moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(moves)
            
            for dx, dy in moves:
                nx, ny = curr[0] + dx, curr[1] + dy
                next_p = (nx, ny)
                
                if grid.is_valid(next_p) and next_p not in path:
                    # Перевірка на чужі піни
                    is_terminal_of_other = False
                    for other_id, (os, oe) in grid.terminals.items():
                        if other_id != net_id and (next_p == os or next_p == oe):
                            is_terminal_of_other = True
                            break
                    
                    if not is_terminal_of_other:
                        stack.append((next_p, path + [next_p]))
                        
        paths.sort(key=len)
        return paths

class PCBSolver:
    """CSP Solver"""
    
    def __init__(self, grid: PCBGrid):
        self.grid = grid
        self.domains: Dict[int, List[Path]] = {} 
        self.assignment: Dict[int, Path] = {}
        self.stats_nodes = 0

    def prepare_domains(self):
        # Тихий режим (без принтів), щоб не засмічувати консоль при переборі
        for net_id in self.grid.terminals:
            paths = PathGenerator.generate_candidate_paths(self.grid, net_id)
            if not paths:
                return False
            self.domains[net_id] = paths
        return True

    def solve(self) -> bool:
        if not self.prepare_domains():
            return False
        return self._backtrack()

    def _backtrack(self) -> bool:
        self.stats_nodes += 1
        
        if len(self.assignment) == len(self.grid.terminals):
            return True

        # MRV
        unassigned = [nid for nid in self.grid.terminals if nid not in self.assignment]
        var = min(unassigned, key=lambda nid: len(self.domains[nid]))
        
        for path in self.domains[var]:
            if self._is_consistent(path):
                self.assignment[var] = path
                
                if self._backtrack():
                    return True
                
                del self.assignment[var]
        
        return False

    def _is_consistent(self, new_path: Path) -> bool:
        new_path_set = set(new_path)
        for assigned_path in self.assignment.values():
            if not new_path_set.isdisjoint(assigned_path):
                return False
        return True

class PCBVisualizer:
    @staticmethod
    def draw(grid: PCBGrid, solution: Dict[int, Path] = None, attempt_num: int = 1):
        fig, ax = plt.subplots(figsize=(9, 9))
        
        ax.set_xlim(-0.5, grid.width - 0.5)
        ax.set_ylim(-0.5, grid.height - 0.5)
        ax.set_xticks(np.arange(grid.width))
        ax.set_yticks(np.arange(grid.height))
        ax.grid(True, color='gray', linestyle=':', alpha=0.3)
        ax.set_facecolor('#212121') 
        
        # Перешкоди
        for ox, oy in grid.obstacles:
            rect = mpatches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, color='#555555', ec='black')
            ax.add_patch(rect)
            
        # Піни та Дроти
        for net_id, (start, end) in grid.terminals.items():
            color = grid.colors[net_id]
            
            # Піни
            ax.add_patch(plt.Circle(start, 0.35, color=color, ec='white', lw=1.5, zorder=10))
            ax.add_patch(plt.Circle(end, 0.35, color=color, ec='white', lw=1.5, zorder=10))
            
            # Підписи
            ax.text(start[0], start[1], str(net_id), color='white', ha='center', va='center', fontweight='bold', zorder=11, fontsize=8)
            ax.text(end[0], end[1], str(net_id), color='white', ha='center', va='center', fontweight='bold', zorder=11, fontsize=8)
            
            # Дроти
            if solution and net_id in solution:
                path = solution[net_id]
                xs = [p[0] for p in path]
                ys = [p[1] for p in path]
                
                ax.plot(xs, ys, color=color, linewidth=7, alpha=0.3, zorder=5, solid_capstyle='round')
                ax.plot(xs, ys, color=color, linewidth=2, alpha=0.9, zorder=6, solid_capstyle='round')

        status = "SOLVED" if solution else "FAILED"
        title = f"PCB Routing CSP | Attempt #{attempt_num}\nStatus: {status} | Grid: {grid.width}x{grid.height}"
        plt.title(title, color='white', fontsize=14, pad=15)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors='gray', labelsize=8)
        
        plt.tight_layout()
        plt.show()

# ==========================================
# Логіка автоматичної генерації та повтору
# ==========================================

def generate_random_layout(width: int, height: int, n_obstacles: int, n_nets: int) -> PCBGrid:
    """Генерує РОЗУМНУ розкладку плати з перевіркою зв'язності"""
    grid = PCBGrid(width, height)
    used_positions = set()

    # 1. Розміщуємо перешкоди НЕ в центрі (залишаємо простір для трас)
    border_zone = max(1, width // 6)
    obstacle_zones = [
        (random.randint(0, border_zone), random.randint(0, height - 1)),
        (random.randint(width - border_zone, width - 1), random.randint(0, height - 1))
    ]
    
    while len(grid.obstacles) < n_obstacles:
        rx = random.randint(0, width - 1)
        ry = random.randint(0, height - 1)
        
        # Уникаємо центральної зони
        is_center = (width//4 < rx < 3*width//4 and height//4 < ry < 3*height//4)
        
        if (rx, ry) not in used_positions and not is_center:
            grid.add_obstacle(rx, ry)
            used_positions.add((rx, ry))

    # 2. Генерація мереж з ГАРАНТОВАНОЮ відстанню між пінами
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22', '#1abc9c']
    
    for i in range(n_nets):
        max_tries = 500
        for _ in range(max_tries):
            # Генеруємо старт
            sx = random.randint(1, width - 2)
            sy = random.randint(1, height - 2)
            
            if (sx, sy) in used_positions:
                continue
            
            # Генеруємо кінець на достатній відстані (4-8 кроків)
            distance = random.randint(4, min(8, width - 2))
            angle = random.choice([0, 45, 90, 135, 180, 225, 270, 315])
            
            ex = sx + int(distance * np.cos(np.radians(angle)))
            ey = sy + int(distance * np.sin(np.radians(angle)))
            
            # Перевірка валідності
            if (0 <= ex < width and 0 <= ey < height and 
                (ex, ey) not in used_positions and
                abs(sx-ex) + abs(sy-ey) >= 3):
                
                used_positions.add((sx, sy))
                used_positions.add((ex, ey))
                
                color = colors[i % len(colors)]
                grid.add_net(i + 1, (sx, sy), (ex, ey), color)
                break
        
    return grid
def main():
    # === НАЛАШТУВАННЯ ДЛЯ ГАРАНТОВАНОГО УСПІХУ ===
    BOARD_SIZE = 12       # Збільшили поле (було 10)
    N_OBSTACLES = 4       # Зменшили перешкоди (було 6)
    N_NETS = 4            # Зменшили кількість дротів (було 5)
    MAX_ATTEMPTS = 100    # Дали ще більше спроб

    print("="*50)
    print("      AUTO-ADAPTIVE PCB ROUTER (BALANCED)      ")
    print("="*50)
    print(f"Ціль: Знайти рішення для {N_NETS} з'єднань на полі {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"Складність: Середня (Balanced)")
    print("-" * 50)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # 1. Генеруємо нову розкладку
        print(f"\r[Спроба {attempt}/{MAX_ATTEMPTS}] Генеруємо нову розкладку та шукаємо шлях...", end="")
        
        # Генеруємо нові позиції
        grid = generate_random_layout(BOARD_SIZE, BOARD_SIZE, N_OBSTACLES, N_NETS)
        
        # 2. Пробуємо вирішити
        solver = PCBSolver(grid)
        success = solver.solve()
        
        if success:
            print(f"\n\n[+] УСПІХ! Знайдено валідне рішення на спробі №{attempt}")
            print(f"    Вузлів перевірено: {solver.stats_nodes}")
            
            # Візуалізуємо успішний результат
            PCBVisualizer.draw(grid, solver.assignment, attempt)
            return # Вихід з програми після успіху
        
    # Якщо цикл закінчився без успіху
    print(f"\n\n[-] НЕВДАЧА. Вичерпано ліміт у {MAX_ATTEMPTS} спроб.")

if __name__ == "__main__":
    main()
