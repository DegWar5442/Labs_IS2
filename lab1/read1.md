# **Pac-Man AI & Pathfinding ᗧ···ᗣ·ᗣ**

A Python implementation of the classic Pac-Man game built with **Pygame**. This project features procedural maze generation, intelligent ghost behaviors, and an autonomous "Auto Mode" where Pac-Man uses pathfinding algorithms to navigate the maze.

## **🎮 Features**

* **Procedural Maze Generation:** Every level features a randomly generated maze, ensuring a unique experience every time.  
* **Dual Control Modes:**  
  * **Manual:** Control Pac-Man yourself using keyboard arrows.  
  * **Automatic (AI):** Watch Pac-Man play autonomously using graph search algorithms to collect pellets and avoid ghosts.  
* **Smart Ghosts:** Ghosts utilize different strategies (Chase vs. Ambush) and use pathfinding to hunt the player.  
* **Progressive Difficulty:** As you advance levels, the maze grows larger, ghosts become faster, and their numbers increase.  
* **Visuals:** Includes sprite-based animations for Pac-Man and ghosts, plus a countdown timer with directional indicators.

## **🛠️ Requirements**

To run this game, you need Python 3 installed along with the following libraries:

* **Pygame:** For graphics and game loop.  
* **NumPy:** For maze array manipulation.

## **📦 Installation**

1. **Clone the repository** or extract the source code.  
2. **Install dependencies:**

pip install pygame numpy

3. Ensure Asset Structure:  
   Make sure you have an assets/ folder in the same directory containing your images:  
   * pacman01.png to pacman03.png  
   * ghost01.png to ghost04.png  
   * wall.png, pacmanDeath.png, arrow.png  
4. **Run the Game:**

python main.py

## **🕹️ Controls**

| Key | Action |
| :---- | :---- |
| **Arrow Keys** | Move Pac-Man manually |
| **H** | Switch to **Auto (AI) Mode** |
| **J** | Switch to **Manual Mode** |
| **Space** | Continue to the next level |
| **Enter** | Select menu options |

## **📂 Project Structure**

The game relies on several modules (based on the imports in main.py):

* **main.py**: The entry point. Handles the game loop, rendering, input, and state management.  
* **maze\_generator.py**: Contains logic (generate\_maze) to create random valid maze layouts.  
* **pathfinding.py**: Implements algorithms like bfs\_search, a\_star\_search to convert the maze into a graph and find paths.  
* **agents.py**: Defines the PacMan and Ghost classes with their movement logic.  
* **heuristics.py**: Contains helper functions for pathfinding (Manhattan/Euclidean distance).  
* **utils.py**: Helper functions for loading images and assets.

## **🧠 AI Logic**

* **Pac-Man (Auto Mode):** Analyzes the graph representation of the maze to find the nearest pellets while maintaining a safe distance from ghosts.  
* **Ghosts:**  
  * **Chase:** Directly targets Pac-Man's current position.  
  * **Ambush:** Attempts to predict Pac-Man's movement or target a nearby location.

*Created with Pygame*