import sys              # used for file reading
from settings import *  # use a separate file for all the constant settings

# the class we will use to store the map, and make calls to path finding
class Grid:
    # set up all the default values for the frid and read in the map from a given file
    def __init__(self, filename):
        # 2D list that will hold all of the grid tile information 
        self.__grid = []
        self.__load_data(filename)
        self.__width, self.__height = len(self.__grid), len(self.__grid[0])

    # loads the grid data from a given file name
    def __load_data(self, filename):
        # turn each line in the map file into a list of integers
        temp_grid = [list(map(int,line.strip())) for line in open(filename, 'r')]
        # transpose the input since we read it in as (y, x) 
        self.__grid = [list(i) for i in zip(*temp_grid)]

    # return the cost of a given action
    # note: this only works for actions in our LEGAL_ACTIONS defined set (8 directions)
    def __get_action_cost(self, action):
        return CARDINAL_COST if (action[0] == 0 or action[1] == 0) else DIAGONAL_COST 

    # returns the tile type of a given position
    def get(self, tile): return self.__grid[tile[0]][tile[1]]
    def width(self):     return self.__width
    def height(self):    return self.__height

    # Student TODO: Implement this function
    # returns true of an object of a given size can navigate from start to goal
    def is_connected(self, start, goal, size):
        return True

    # Student TODO: Replace this function with your A* implementation
    # returns a sample path from start tile to end tile which is probably illegal
    def get_path(self, start, end, size):
        path = []
        action = (1 if start[0] <= end[0] else -1, 1 if start[1] <= end[1] else -1)
        d = (abs(start[0] - end[0]), abs(start[1] - end[1]))
        # add the diagonal actions until we hit the row or column of the end tile
        for diag in range(d[1] if d[0] > d[1] else d[0]):
            path.append(action)
        # add the remaining straight actions to reach the end tile
        for straight in range(d[0]-d[1] if d[0] > d[1] else d[1]-d[0]):
            path.append((action[0], 0) if d[0]>d[1] else (0, action[1]))
        # return the path, the cost of the path, and the set of expanded nodes (for A*)
        return path, sum(map(self.__get_action_cost, path)), set()

    # Student TODO: Replace this function with a better (but admissible) heuristic
    # estimate the cost for moving between start and end
    def estimate_cost(self, start, goal):
        return 1


# Student TODO: You should implement AStar as a separate class
#               This will help keep things modular

# Student TODO: You should implement a separate Node class
#               AStar search should use these Nodes in its open and closed lists
