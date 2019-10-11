import os, subprocess, time, signal, random, sys
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import numpy as np

class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    player_marker = "X"
    objective_marker = "O" #2
    wall_marker = "*" #1
    free_square = ' ' #0
    hole_square = 'H' #3
    initialized = False
    maze_dimensions = None
    max_steps = None
    walls_door = None
    player_start_pos = None
    player_marker_position_standard = [1, 1]
    player_start_pos_random = 'random'
    player_start_pos_standard = 'standard'

    def __init__(self):
        self.action_space = spaces.Discrete(4) #Su, giu, destra, sinistra
        # obs = <posizione, squares_around_pos>
        self.observation_space = spaces.Box(np.array([np.iinfo(np.uint8).min, np.iinfo(np.uint8).min, np.iinfo(np.uint8).min, np.iinfo(np.uint8).min,
                                                      np.iinfo(np.uint8).min, np.iinfo(np.uint8).min]), np.array([np.iinfo(np.uint8).max, np.iinfo(np.uint8).max,
                                                      np.iinfo(np.uint8).max, np.iinfo(np.uint8).max, np.iinfo(np.uint8).max, np.iinfo(np.uint8).max]), dtype=np.uint8)
        self.player_marker_start_pos = None
        self.player_marker_position = [None, None]

    def initialize_env(self, dimensions = 10, max_steps = 2000, walls_door = 2, player_start_pos = 'standard'):
        assert dimensions >= 5, "La dimensione minima consentita e' 5."
        assert dimensions <= 250, "La dimensione massima consentita e' 250."
        assert max_steps > 100, "Il minimo numero di passi consentito e' 100."
        assert walls_door >= 1, "Il numero minimo di porte consentito e' 1."
        assert walls_door <= dimensions - 2, "Il massimo numero di porte consentite e' dimensione - 2."
        assert player_start_pos == self.player_start_pos_standard or player_start_pos == self.player_start_pos_random, "Valori validi per player_start_pos sono standard o random"
        self.maze_dimensions = dimensions
        self.max_steps = max_steps
        self.walls_door = walls_door
        self.player_start_pos = player_start_pos
        self.maze = np.array(self.create_maze(self.maze_dimensions))
        self.objective_position = self.add_objective_marker(None)
        self.player_marker_start_pos = tuple(self.add_player_marker(None))
        self.player_marker_position[0] = self.player_marker_start_pos[0]
        self.player_marker_position[1] = self.player_marker_start_pos[1]
        self.num_movements = 0
        self.initialized = True

    def step(self, action):
        '''
        0: go up
        1: go down
        2: go left
        3: go right
        '''
        assert self.action_space.contains(action), "Errore! Azione non riconosciuta"
        self.num_movements += 1
        current_pos_x = self.player_marker_position[0]
        current_pos_y = self.player_marker_position[1]
        new_pos_y = current_pos_y
        new_pos_x = current_pos_x
        next_square = None
        reward = 0
        costant_decay = -0.1
        done = False
        if (action == 0):
            reward = costant_decay * self.num_movements
            next_square = self.maze[current_pos_y - 1][current_pos_x]
            if (next_square == self.wall_marker):
                reward = reward + costant_decay
            elif (next_square == self.free_square):
                new_pos_y = current_pos_y - 1
                new_pos_x = current_pos_x
                self.maze[current_pos_y][current_pos_x] = self.free_square
                self.maze[new_pos_y][new_pos_x] = self.player_marker
            elif(next_square == self.objective_marker):
                done = True
                reward = reward + 30
        elif (action == 1):
            reward = costant_decay * self.num_movements
            next_square = self.maze[current_pos_y + 1][current_pos_x]
            if (next_square == self.wall_marker):
                reward = reward + costant_decay
            elif (next_square == self.free_square):
                new_pos_y = current_pos_y + 1
                new_pos_x = current_pos_x
                self.maze[current_pos_y][current_pos_x] = self.free_square
                self.maze[new_pos_y][new_pos_x] = self.player_marker
            elif (next_square == self.objective_marker):
                done = True
                reward = reward + 30
        elif (action == 2):
            reward = costant_decay * self.num_movements
            next_square = self.maze[current_pos_y][current_pos_x - 1]
            if (next_square == self.wall_marker):
                reward = reward + costant_decay
            elif (next_square == self.free_square):
                new_pos_y = current_pos_y
                new_pos_x = current_pos_x - 1
                self.maze[current_pos_y][current_pos_x] = self.free_square
                self.maze[new_pos_y][new_pos_x] = self.player_marker
            elif (next_square == self.objective_marker):
                done = True
                reward = reward + 30
        elif (action == 3):
            reward = costant_decay * self.num_movements
            next_square = self.maze[current_pos_y][current_pos_x + 1]
            if (next_square == self.wall_marker):
                reward = reward + costant_decay
            elif (next_square == self.free_square):
                new_pos_y = current_pos_y
                new_pos_x = current_pos_x + 1
                self.maze[current_pos_y][current_pos_x] = self.free_square
                self.maze[new_pos_y][new_pos_x] = self.player_marker
            elif (next_square == self.objective_marker):
                done = True
                reward = reward + 30

        self.player_marker_position[0] = new_pos_x
        self.player_marker_position[1] = new_pos_y
        y_pos = self.player_marker_position[1]
        x_pos = self.player_marker_position[0]
        up_square = self.get_square_type(self.maze[y_pos - 1][x_pos])
        down_square = self.get_square_type(self.maze[y_pos + 1][x_pos])
        left_square = self.get_square_type(self.maze[y_pos][x_pos - 1])
        right_square = self.get_square_type(self.maze[y_pos][x_pos + 1])
        self.print_maze()
        if(self.num_movements > self.max_steps):
            reward = -30
            done = True
            return [x_pos, y_pos, up_square, down_square, left_square, right_square], reward, done, {"num_steps": self.num_movements}
        else:
            return [x_pos, y_pos, up_square, down_square, left_square, right_square], reward, done, {"num_steps": self.num_movements}

    '''Reset the enviroment but the maze is the same. Keep the q-table'''
    def reset(self):
        assert self.initialized is True, "Initialize env with initilize_env() function before call reset() method."
        self.num_movements = 0
        #Clean position
        self.maze[self.player_marker_position[1]][self.player_marker_position[0]] = self.free_square
        self.player_marker_position = self.add_player_marker([self.player_marker_start_pos[0],self.player_marker_start_pos[1]])
        self.print_maze()
        y_pos = self.player_marker_position[1]
        x_pos = self.player_marker_position[0]
        up_square = self.get_square_type(self.maze[y_pos - 1][x_pos])
        down_square = self.get_square_type(self.maze[y_pos + 1][x_pos])
        left_square = self.get_square_type(self.maze[y_pos][x_pos - 1])
        right_square = self.get_square_type(self.maze[y_pos][x_pos + 1])
        return [x_pos, y_pos, up_square, down_square, left_square, right_square], 0, False, {"num_steps": self.num_movements}

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def create_maze(self, dimensions):

        def add_door(maze, dimensions, col_start):
            dimensions = dimensions - 1
            if (col_start % 2 != 0):
                for col in range(col_start, dimensions):
                    if(col % 2 != 0):
                        door_count = 0
                        while(door_count < self.walls_door):
                            for _ in range(0, self.walls_door):
                                i = random.randint(1, dimensions - 1)
                                j = col
                                if(maze[i][j] == self.wall_marker):
                                    maze[i][j] = self.free_square
                                    door_count += 1
            else:
                for col in range(col_start, dimensions):
                    if (col % 2 == 0):
                        door_count = 0
                        while (door_count < self.walls_door):
                            for _ in range(0, self.walls_door):
                                i = random.randint(1, dimensions - 1)
                                j = col
                                if (maze[i][j] == self.wall_marker):
                                    maze[i][j] = self.free_square
                                    door_count += 1
            return maze

        def add_wall_inside(maze, dimensions, col_start):
            dimensions = dimensions - 1
            if(col_start % 2 != 0):
                for col in range(col_start, dimensions):
                    for row in range(1, dimensions):
                        if(col % 2 != 0):
                            maze[row][col] = self.wall_marker
            else:
                for col in range(col_start, dimensions):
                    for row in range(1, dimensions):
                        if (col % 2 == 0):
                            maze[row][col] = self.wall_marker
            return maze

        maze = []
        for i in range(0, dimensions):
            row = []
            if(i == 0 or i == dimensions - 1):
                for j in range(0, dimensions):
                    row.append(self.wall_marker)
                maze.append(row)
            else:
                for j in range(0, dimensions):
                    if(j == 0 or j == dimensions-1):
                        row.append(self.wall_marker)
                    else:
                        row.append(self.free_square)
                maze.append(row)
        col_start = random.randint(1,2)
        maze = add_wall_inside(maze, dimensions, col_start)
        maze = add_door(maze, dimensions, col_start)
        return maze

    def add_player_marker(self, marker_position):
        pos = marker_position
        if(marker_position is not None):
            if((marker_position[0] > self.maze.shape[0] - 1) or (marker_position[0] < 1)):
                print("Posizione Marker non valida")
                sys.exit(0)
            elif((marker_position[1] > self.maze.shape[0] - 1) or (marker_position[1] < 1)):
                print("Posizione Marker non valida")
                sys.exit(0)
            else:
                self.maze[marker_position[1]][marker_position[0]] = self.player_marker
        else:
            if(self.player_start_pos == self.player_start_pos_standard):
                self.maze[self.player_marker_position_standard[1]][self.player_marker_position_standard[0]] = self.player_marker
                pos = [self.player_marker_position_standard[0], self.player_marker_position_standard[1]]
            elif(self.player_start_pos == self.player_start_pos_random):
                i = random.randint(1, self.maze.shape[0] - 2)
                j = random.randint(1, self.maze.shape[0] - 2)
                self.maze[i][j] = self.player_marker
                pos = [j,i]
            else:
                self.maze[self.player_marker_position_standard[1]][self.player_marker_position_standard[0]] = self.player_marker
                pos = [self.player_marker_position_standard[0], self.player_marker_position_standard[1]]
        return pos

    def add_objective_marker(self, objective_pos):

        def find_all_free_pos(maze):
            dimensions = maze.shape[0]
            list_of_free_sqaure = []
            for i in range(0, dimensions):
                for j in range(0, dimensions):
                    if(maze[i][j] == self.free_square):
                        list_of_free_sqaure.append([i,j])
            return  list_of_free_sqaure

        if(objective_pos is None):
            free_square_list = find_all_free_pos(self.maze)
            free_square = random.choice(free_square_list)
            i = free_square[0]
            j = free_square[1]
            self.maze[i][j] = self.objective_marker
            return free_square
        else:
            if (self.maze[objective_pos[1]][objective_pos[0]] == self.wall_marker):
                print("Obiettivo non aggiunto")
                sys.exit(0)
            elif (self.maze[objective_pos[1]][objective_pos[0]] == self.player_marker):
                print("Obiettivo non aggiunto")
                sys.exit(0)
            else:
                self.maze[objective_pos[1]][objective_pos[0]] = self.objective_marker
            return objective_pos

    def get_square_type(self, sym):
        val = None
        if(sym == self.free_square):
            val = 0
        elif(sym == self.wall_marker):
            val = 1
        elif (sym == self.objective_marker):
            val = 2
        elif (sym == self.hole_square):
            val = 3
        return val

    def print_maze(self):
        os.system('clear')
        dimensions = self.maze.shape
        y_dim = dimensions[0]
        x_dim = dimensions[1]
        for y in range(0, y_dim):
            for x in range(0, x_dim):
                print (self.maze[y][x], end=' ')
            print ("\n", end='\r')

    def get_action_space(self):
        return self.action_space
