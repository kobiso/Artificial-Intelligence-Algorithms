import sys  # used for file reading
from settings import *  # use a separate file for all the constant settings
import numpy as np
from heapq import *


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
        self.path = []

    # loads the grid data from a given file name
    def __load_data(self, filename):
        # turn each line in the map file into a list of integers
        temp_grid = [list(map(int, line.strip())) for line in open(filename, 'r')]
        # transpose the input since we read it in as (y, x)
        self.__grid = [list(i) for i in zip(*temp_grid)]

    # return the cost of a given action
    # note: this only works for actions in our LEGAL_ACTIONS defined set (8 directions)
    def __get_action_cost(self, action):
        return CARDINAL_COST if (action[0] == 0 or action[1] == 0) else DIAGONAL_COST

    # returns the tile type of a given position
    def get(self, tile):
        return self.__grid[tile[0]][tile[1]]

    def width(self):
        return self.__width

    def height(self):
        return self.__height

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

    # returns true of an object of a given size can navigate from start to goal
    def is_connected(self, start, goal, size):
        if self.sectors[size][start[0]][start[1]] != 0 and self.sectors[size][goal[0]][goal[1]] != 0 and \
                        self.sectors[size][start[0]][start[1]] == self.sectors[size][goal[0]][goal[1]]:
            return True
        return False

    # returns a sample path from start tile to end tile which is probably illegal
    def get_path_WRONG(self, start, end, size):
        path = []
        action = (1 if start[0] <= end[0] else -1, 1 if start[1] <= end[1] else -1)
        d = (abs(start[0] - end[0]), abs(start[1] - end[1]))
        # add the diagonal actions until we hit the row or column of the end tile
        for diag in range(d[1] if d[0] > d[1] else d[0]):
            path.append(action)
        # add the remaining straight actions to reach the end tile
        for straight in range(d[0] - d[1] if d[0] > d[1] else d[1] - d[0]):
            path.append((action[0], 0) if d[0] > d[1] else (0, action[1]))
        # return the path, the cost of the path, and the set of expanded nodes (for A*)
        return path, sum(map(self.__get_action_cost, path)), set()

    def get_path(self, start, end, size):
        return self.AStarClass.AStarAlgorithm(start, end, size)

    # TODO: Replace this function with a better (but admissible) heuristic
    # estimate the cost for moving between start and end
    def estimate_cost(self, start, goal):
        return np.sqrt(np.square(start[0] - goal[0]) + np.square(start[1] - goal[
            1])) * CARDINAL_COST  # Calculating 'Euclidean Distance' of the grid for admissible heuristic cost

class AStar:
    def __init__(self, Grid):
        self.grid = Grid
        self.nodes = []
        for i in range(self.grid.width()):
            self.nodes.append([None] * self.grid.height())

    def reset(self):
        self.path = []
        self.cost = 0
        for i in range(self.grid.width()):
            for j in range(self.grid.height()):
                self.nodes[i][j] = Node((i, j))
        self.closedSet = set()  # Use set for closed node
        self.openList = []  # Use numpy array maybe?

    def calculate_path_and_cost(self, node, start):
        if node.state == start:
            self.path.reverse()
            return self.path, self.cost
        else:
            self.path.append(node.action)
            self.cost += self.action_cost(node.action[0], node.action[1])
            return self.calculate_path_and_cost(node.parent, start)

    def expand(self, node, size):
        expandable = []
        for index in LEGAL_ACTIONS:
            try:
                expandNode = (node.state[0] + index[0], node.state[1] + index[1])
                if expandNode[0] < 0 or expandNode[1] < 0:
                    continue
                if self.grid.is_connected(node.state, expandNode, size):
                    if self.grid.is_connected(node.state, (node.state[0], expandNode[1]), size) and \
                            self.grid.is_connected(node.state, (expandNode[0], node.state[1]), size):
                        expandable.append(index)
                    else:
                        continue
            except IndexError:
                continue
        return expandable

    def action_cost(self, xAxis, yAxis):
        if abs(xAxis + yAxis) == 1:
            return CARDINAL_COST
        else:
            return DIAGONAL_COST

    def AStarAlgorithm(self, start, goal, size):
        self.reset()
        if not self.grid.is_connected(start, goal, size):
            return self.path, self.cost, self.closedSet

        heappush(self.openList, self.nodes[start[0]][start[1]])
        while self.openList:
            # heapify(self.openList)
            node = heappop(self.openList)
            self.closedSet.add(node.state)
            if node.state == goal:
                self.path, self.cost = self.calculate_path_and_cost(node, start)
                return self.path, self.cost, self.closedSet
            for exp in self.expand(node, size):
                child = self.nodes[node.state[0] + exp[0]][node.state[1] + exp[1]]
                if child.state in self.closedSet:
                    continue
                child_f = node.g + self.action_cost(exp[0], exp[1])
                if child in self.openList and child.g <= child_f:
                    continue
                child.update(node, exp, child_f, child_f + self.grid.estimate_cost(child.state, goal))
                if child not in self.openList:
                    heappush(self.openList, child)
        return self.path, self.cost, self.closedSet

#               AStar search should use these Nodes in its open and closed lists
class Node:
    def __init__(self, tile):
        self.state = tile
        self.action = (0, 0)
        self.g = 0
        self.f = 0
        self.parent = None

    def update(self, parent, action, g, f):
        self.parent = parent
        self.action = action
        self.g = g
        self.f = f

    def __lt__(self, other):
        if self.f == other.f:
            return self.g < other.g
        return self.f < other.f

    def __eq__(self, other):
        return self.state == other.state
