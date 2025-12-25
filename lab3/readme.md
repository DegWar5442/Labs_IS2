# **Mobile Network Planning with Genetic Algorithms 📡**

This project implements a **Genetic Algorithm (GA)** to solve the **Mobile Network Planning Problem**. It optimizes the placement and type of telecommunication towers to maximize user coverage while minimizing infrastructure costs and adhering to strict geographical constraints.

## **🎯 Objective**

Find the optimal configuration of towers (position and type) to cover 60+ users scattered across a 100x100 km area, balancing:

* **Cost Efficiency:** Minimizing equipment and installation costs.  
* **Coverage Quality:** Maximizing the percentage of connected users.  
* **Constraints:** Respecting capacity limits, terrain restrictions, and infrastructure proximity.

## **🛠 Features**

### **1\. Heterogeneous Network**

The simulation supports different tower types with distinct characteristics:

| Type | Name | Cost | Radius | Capacity | Restrictions |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Type 1** | **Small 4G** | $800 | 15 km | **8 users** | None (Can be built anywhere) |
| **Type 2** | **Big 5G** | $2,000 | 40 km | **30 users** | Must be within **10km** of a road. Cannot be built on unstable ground. |

### **2\. Advanced Constraints**

* **Infrastructure Dependency:** Big towers require roads for heavy equipment access. If placed far from a road, a massive penalty is applied (simulating the cost of building a new access road).  
* **Terrain Analysis:** Certain areas contain **Unstable Ground** (swamps/sand). Heavy 5G towers are forbidden here, forcing the algorithm to use lighter 4G towers or avoid the area.  
* **Capacity Limits:** Each tower has a hard limit on connected users. Users automatically connect to the nearest *available* tower. If all nearby towers are full, the user remains uncovered.

### **3\. Visualization**

The script generates a dual-view dashboard:

* **Left Panel:** A tactical map showing towers, roads, unstable zones, and user connections.  
  * **Green Circles:** Small 4G coverage.  
  * **Purple Circles:** Big 5G coverage.  
  * **Red Markers:** Overloaded towers or users without signal.  
* **Right Panel:** Evolution graphs showing Coverage (%) vs. Cost ($) over generations.

## **🚀 Getting Started**

### **Prerequisites**

You need Python installed along with the matplotlib library for visualization.

pip install matplotlib

### **Running the Simulation**

Simply run the script:

python mobile\_towers\_ga.py

The algorithm will run for 200 generations (configurable) and display the final result window.

## **🧬 Algorithm Details**

* **Chromosome:** Integer string where each gene represents a potential site state (0=Empty, 1=4G, 2=5G).  
* **Fitness Function:** Minimize Total Cost \+ Uncovered Penalties \+ Constraint Penalties.  
* **Selection:** Tournament selection.  
* **Crossover:** Single-point crossover.  
* **Mutation:** Randomly changing tower type at a specific site.

## **📂 Project Structure**

.  
├── mobile\_towers\_ga.py    \# Main source code containing the GA logic and visualization  
└── README.md              \# Project documentation

