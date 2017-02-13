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
        self.compute_sectors()
        self.AStarClass = AStar(self)
        self.path =[]

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

    # TODO: Implement ComputeSectors and flood_fill method
    def compute_sectors(self):
        self.sectors = np.zeros((MAX_SIZE + 1, self.width(), self.height()))
        self.sector_num = np.zeros(4)
        for size in range(1, MAX_SIZE + 1):
            for x in range(self.width()):
                for y in range(self.height()):
                    self.sector_num[size] += 1
                    self.flood_fill((x, y), self.get((x, y)), size)

    def flood_fill(self, pos, type, size):
        try:
            if self.sectors[size][pos[0]][pos[1]] != 0: return
            for i in range(size):
                for j in range(size):
                    if self.get((pos[0] + i, pos[1] + j)) != type: return
        except IndexError:
            return
        self.sectors[size][pos[0]][pos[1]] = self.sector_num[size]
        self.flood_fill((pos[0] + 1, pos[1]), type, size)
        self.flood_fill((pos[0] - 1, pos[1]), type, size)
        self.flood_fill((pos[0], pos[1] + 1), type, size)
        self.flood_fill((pos[0], pos[1] - 1), type, size)