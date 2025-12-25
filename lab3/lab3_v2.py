import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
from matplotlib.lines import Line2D # Потрібно для легенди

# ==========================================
# 1. Налаштування симуляції
# ==========================================
AREA_SIZE = 100
NUM_USERS = 1000
NUM_POSSIBLE_TOWERS = 100
PENALTY_UNCOVERED = 100000

# Типи веж
# 0: Нічого
# 1: Мала вежа (4G)
# 2: Велика вежа (5G)
TOWER_TYPES = {
    0: {'cost': 0,    'rad': 0,  'cap': 0,  'color': 'none'},
    1: {'cost': 800,  'rad': 15, 'cap': 8,  'color': 'green', 'name': 'Small 4G (Max 8)'},
    2: {'cost': 2000, 'rad': 40, 'cap': 30, 'color': 'purple', 'name': 'Big 5G (Max 30)'}
}

# Обмеження
MAX_DIST_TO_ROAD = 10
PENALTY_INFRASTRUCTURE = 3000
PENALTY_UNSTABLE_GROUND = 10000

ROADS = [
    [10, 10, 90, 90],
    [10, 90, 90, 10],
    [50, 0, 50, 100]
]

UNSTABLE_ZONES = [
    [60, 60, 30, 25],
    [15, 35, 20, 20]
]

# Параметри ГА
POP_SIZE = 100
GENERATIONS = 150
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
ELITE_SIZE = 5

def dist_point_to_segment(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
    t = max(0, min(1, t))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

def is_in_rect(px, py, rect):
    rx, ry, rw, rh = rect
    return rx <= px <= rx + rw and ry <= py <= ry + rh

class NetworkOptimizationGA:
    def __init__(self):
        # Генерація абонентів
        self.users = []
        for _ in range(NUM_USERS):
            if random.random() < 0.7: 
                self.users.append((random.gauss(50, 15), random.gauss(50, 15)))
            else:
                self.users.append((random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE)))
        self.users = [(max(0, min(AREA_SIZE, x)), max(0, min(AREA_SIZE, y))) for x, y in self.users]

        self.tower_sites = []
        for _ in range(NUM_POSSIBLE_TOWERS):
            self.tower_sites.append((random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE)))

        self.site_info = []
        for tx, ty in self.tower_sites:
            min_dist = float('inf')
            for r in ROADS:
                d = dist_point_to_segment(tx, ty, r[0], r[1], r[2], r[3])
                if d < min_dist: min_dist = d
            
            in_unstable = False
            for zone in UNSTABLE_ZONES:
                if is_in_rect(tx, ty, zone):
                    in_unstable = True
                    break
            self.site_info.append({'dist_road': min_dist, 'in_unstable': in_unstable})

    # === ЕВРИСТИКА 1: Розумна Ініціалізація ===
    def heuristic_initialization(self):
        genome = []
        for tx, ty in self.tower_sites:
            users_nearby = sum(1 for ux, uy in self.users if math.sqrt((tx-ux)**2 + (ty-uy)**2) < 30)
            if users_nearby >= 10:
                choice = random.choices([0, 1, 2], weights=[0.1, 0.3, 0.6])[0]
            elif users_nearby >= 3:
                choice = random.choices([0, 1, 2], weights=[0.3, 0.6, 0.1])[0]
            else:
                choice = random.choices([0, 1, 2], weights=[0.9, 0.09, 0.01])[0]
            genome.append(choice)
        return genome

    def calculate_metrics(self, individual):
        cost = 0
        penalty = 0
        tower_loads = {} 
        
        for i, t_type in enumerate(individual):
            if t_type == 0: continue
            tower_loads[i] = 0
            cost += TOWER_TYPES[t_type]['cost']
            
            if t_type == 2:
                if self.site_info[i]['dist_road'] > MAX_DIST_TO_ROAD:
                    penalty += PENALTY_INFRASTRUCTURE + (self.site_info[i]['dist_road'] * 100)
                if self.site_info[i]['in_unstable']:
                    penalty += PENALTY_UNSTABLE_GROUND

        covered_users_indices = set()
        
        for u_idx, (ux, uy) in enumerate(self.users):
            candidates = []
            for i, t_type in enumerate(individual):
                if t_type == 0: continue
                rad = TOWER_TYPES[t_type]['rad']
                tx, ty = self.tower_sites[i]
                dist = math.sqrt((tx - ux)**2 + (ty - uy)**2)
                
                if dist <= rad:
                    candidates.append((dist, i))
            
            candidates.sort(key=lambda x: x[0])
            
            is_connected = False
            for dist, t_idx in candidates:
                t_type = individual[t_idx]
                cap = TOWER_TYPES[t_type]['cap']
                
                if tower_loads[t_idx] < cap:
                    tower_loads[t_idx] += 1
                    is_connected = True
                    break 
            
            if is_connected:
                covered_users_indices.add(u_idx)
        
        num_uncovered = NUM_USERS - len(covered_users_indices)
        penalty += num_uncovered * PENALTY_UNCOVERED
        
        coverage_pct = (len(covered_users_indices) / NUM_USERS) * 100
        total_loss = cost + penalty
        
        return -total_loss, cost, coverage_pct, penalty, tower_loads

    def calculate_fitness(self, individual):
        fit, _, _, _, _ = self.calculate_metrics(individual)
        return fit

    def crossover(self, parent1, parent2):
        if random.random() > CROSSOVER_RATE:
            return parent1[:], parent2[:]
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

    # === ЕВРИСТИКА 2: Розумна Мутація ===
    def heuristic_mutate(self, individual):
        _, _, _, _, loads = self.calculate_metrics(individual)
        
        for i in range(len(individual)):
            if random.random() < MUTATION_RATE:
                current_type = individual[i]
                current_load = loads.get(i, 0)
                
                if current_type > 0 and current_load == 0:
                    individual[i] = 0
                elif current_type == 1 and current_load >= 8:
                    individual[i] = 2
                elif current_type == 2 and current_load <= 5:
                    individual[i] = 1
                else:
                    individual[i] = random.choice([0, 0, 1, 2])
        return individual

    def run(self):
        population = [self.heuristic_initialization() for _ in range(POP_SIZE)]
        
        best_solution = None
        best_fitness = -float('inf')
        best_loads = {}
        
        history_cost = []
        history_coverage = []

        print(f"Старт симуляції з евристиками...")

        for gen in range(GENERATIONS):
            fitnesses = []
            for ind in population:
                fit = self.calculate_fitness(ind)
                fitnesses.append(fit)
            
            gen_max = max(fitnesses)
            best_idx = fitnesses.index(gen_max)
            
            _, cost, cov, _, loads = self.calculate_metrics(population[best_idx])
            history_cost.append(cost)
            history_coverage.append(cov)

            if gen_max > best_fitness:
                best_fitness = gen_max
                best_solution = copy.deepcopy(population[best_idx])
                best_loads = loads
            
            selected = self.tournament_selection(population, fitnesses)
            next_pop = []
            
            zipped = sorted(zip(fitnesses, population), reverse=True)
            next_pop.extend([copy.deepcopy(ind) for fit, ind in zipped[:ELITE_SIZE]])
            
            while len(next_pop) < POP_SIZE:
                p1 = random.choice(selected)
                p2 = random.choice(selected)
                c1, c2 = self.crossover(p1, p2)
                next_pop.append(self.heuristic_mutate(c1))
                if len(next_pop) < POP_SIZE:
                    next_pop.append(self.heuristic_mutate(c2))
            
            population = next_pop

        self.visualize(best_solution, history_cost, history_coverage, best_loads)

    def tournament_selection(self, population, fitnesses, k=3):
        selected = []
        for _ in range(POP_SIZE):
            candidates_idx = random.sample(range(POP_SIZE), k)
            winner_idx = max(candidates_idx, key=lambda i: fitnesses[i])
            selected.append(population[winner_idx])
        return selected

    def visualize(self, solution, hist_cost, hist_cov, loads):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # === 1. КАРТА (Відновлена детальна візуалізація) ===
        # Зони
        for z in UNSTABLE_ZONES:
            rect = patches.Rectangle((z[0], z[1]), z[2], z[3], linewidth=1, edgecolor='brown', facecolor='orange', alpha=0.3)
            ax1.add_patch(rect)

        # Дороги
        for i, r in enumerate(ROADS):
            ax1.plot([r[0], r[2]], [r[1], r[3]], color='gray', linewidth=4, alpha=0.5)

        # Вежі
        for i, t_type in enumerate(solution):
            if t_type == 0: continue
            
            tx, ty = self.tower_sites[i]
            params = TOWER_TYPES[t_type]
            
            # Коло покриття
            circle = patches.Circle((tx, ty), params['rad'], color=params['color'], alpha=0.1)
            ax1.add_patch(circle)
            
            # Маркер вежі + Текст завантаження
            marker = 'o' if t_type == 1 else 'D'
            current_load = loads.get(i, 0)
            max_cap = params['cap']
            
            # Якщо вежа повна - червоний колір
            if current_load >= max_cap:
                ax1.plot(tx, ty, marker, color='red', markersize=10)
            else:
                ax1.plot(tx, ty, marker, color=params['color'], markersize=8)
            
            ax1.text(tx, ty+2, f"{current_load}/{max_cap}", fontsize=8, color='black', ha='center')

            # Перевірка помилок (для візуалізації)
            is_error = False
            if t_type == 2 and self.site_info[i]['dist_road'] > MAX_DIST_TO_ROAD: is_error = True
            if t_type == 2 and self.site_info[i]['in_unstable']: is_error = True
            if is_error: ax1.plot(tx, ty, 'rx', markersize=12, markeredgewidth=2)

        # Абоненти (Перерахунок для коректного відображення червоних хрестиків)
        # Нам треба знати, хто підключився у кращому рішенні
        temp_loads = {i:0 for i in loads} 
        for u_idx, (ux, uy) in enumerate(self.users):
            candidates = []
            for i, t_type in enumerate(solution):
                if t_type == 0: continue
                rad = TOWER_TYPES[t_type]['rad']
                tx, ty = self.tower_sites[i]
                dist = math.sqrt((tx - ux)**2 + (ty - uy)**2)
                if dist <= rad: candidates.append((dist, i))
            candidates.sort(key=lambda x: x[0])
            
            is_conn = False
            for dist, t_idx in candidates:
                limit = TOWER_TYPES[solution[t_idx]]['cap']
                if temp_loads[t_idx] < limit:
                    temp_loads[t_idx] += 1
                    is_conn = True
                    break
            
            if is_conn:
                ax1.plot(ux, uy, '.', color='blue', alpha=0.6)
            else:
                ax1.plot(ux, uy, 'x', color='red', alpha=0.8)

        # Легенда
        custom_lines = [
            Line2D([0], [0], color='green', marker='o', linestyle='None', label='Small 4G (Max 8)'),
            Line2D([0], [0], color='purple', marker='D', linestyle='None', label='Big 5G (Max 30)'),
            Line2D([0], [0], color='red', marker='o', linestyle='None', label='Tower Full'),
            Line2D([0], [0], color='blue', marker='.', linestyle='None', label='Connected'),
            Line2D([0], [0], color='red', marker='x', linestyle='None', label='No Signal')
        ]
        ax1.legend(handles=custom_lines, loc='upper right')
        
        ax1.set_title(f"Карта (Load/Capacity)\nПокриття: {hist_cov[-1]:.1f}% | Ціна: ${hist_cost[-1]}")
        ax1.set_xlim(0, AREA_SIZE)
        ax1.set_ylim(0, AREA_SIZE)
        ax1.grid(True, linestyle=':', alpha=0.4)

        # === 2. ГРАФІК ДИНАМІКИ ===
        color = 'tab:blue'
        ax2.set_xlabel('Покоління')
        ax2.set_ylabel('Покриття (%)', color=color)
        ax2.plot(hist_cov, color=color, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 105)
        ax2.grid(True)

        ax2_right = ax2.twinx()  
        color = 'tab:red'
        ax2_right.set_ylabel('Ціна ($)', color=color)  
        ax2_right.plot(hist_cost, color=color, linestyle='--')
        ax2_right.tick_params(axis='y', labelcolor=color)

        ax2.set_title("Динаміка: Покриття vs Ціна")
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    ga = NetworkOptimizationGA()
    ga.run()