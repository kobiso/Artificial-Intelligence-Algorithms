import sys              # used for file reading
import time             # used for timing the path-finding
import pygame as pg     # used for drawing / event handling
from settings import *  # use a separate file for all the constant settings
import grid_algo        # grid class

class PathFindingDisplay:
    
    def __init__(self, mapfile):
        # initialize pygame library things
        pg.init()
        pg.display.set_caption(TITLE)
        pg.key.set_repeat(500, 100)
        # load the particular Grid class that we want to test
        self.__grid       = grid_algo.Grid(mapfile)
        # declare the remaining variables we'll need for the GUI
        self.__width      = self.__grid.width() * TILE_SIZE
        self.__height     = self.__grid.height() * TILE_SIZE
        self.__screen     = pg.display.set_mode((self.__width, self.__height))
        self.__font       = pg.font.SysFont(FONT_NAME, FONT_SIZE, bold=FONT_BOLD)
        self.__clock      = pg.time.Clock()
        self.__m_down     = [False]*3 # which mouse button is being held down    
        self.__m_tile     = (-1, -1)  # tile the mouse is currently on
        self.__s_tile     = (-1, -1)  # start tile for path finding
        self.__o_size     = 1         # the size of the object to pathfind
        self.__path       = []        # previous computed path
        self.__path_cost  = 0         # previous computed path cost
        self.__expanded   = set()     # previous computed set of nodes expanded
        self.__path_time  = 0         # previous path computation time
        self.__heuristic  = 0         # previous computed heuristic value                       
        # prerender the map surface, which will never change for a static map
        self.__map_surface = pg.Surface((self.__width, self.__height))
        for x in range(self.__grid.width()):
            for y in range(self.__grid.height()):
                self.__draw_tile(self.__map_surface, (x,y), TILE_COLOR[self.__grid.get((x,y))], 1)

    # game main loop update function
    def update(self):
        self.__events()             # handle all mouse and keyboard events
        self.__compute_path()       # compute a path if we need to
        self.__draw()               # draw everything to the screen

    # draw everything to the screen
    def __draw(self):
        self.__screen.blit(self.__map_surface, (0,0))
        self.__draw_connected()
        self.__draw_path()
        self.__draw_grid_lines()
        self.__draw_stats()
        pg.display.flip()

    # draw the all the tiles that are connected (path possible) from the mouseover tile
    def __draw_connected(self):
        if self.__m_down[2]: # only draw if we're holding down the right mouse button
            for x in range(self.__grid.width()):
                for y in range(self.__grid.height()):
                    if self.__grid.is_connected(self.__m_tile, (x, y), self.__o_size):
                        self.__draw_tile(self.__screen, (x, y), PURPLE, self.__o_size)

    # draw the path and expanded tiles
    def __draw_path(self):
        # draw all the tiles that were expanded by the pathfinding search
        for tile in self.__expanded:
            self.__draw_tile(self.__screen, tile, RED, self.__o_size)
        # draw nodes from the satrt tile following the path to the goal tile
        current = self.__s_tile[:]
        for action in self.__path:
            self.__draw_tile(self.__screen, current, WHITE, self.__o_size)
            current = (current[0] + action[0], current[1] + action[1])
        # draw the path start and end tiles
        self.__draw_tile(self.__screen, self.__m_tile, YELLOW, self.__o_size)
        if self.__s_tile != (-1, -1): self.__draw_tile(self.__screen, self.__s_tile, YELLOW, self.__o_size)

    # draw some statistics to the screen
    def __draw_stats(self):
        self.__draw_text(str(self.__s_tile) + " " + str(self.__m_tile), (10, self.__height-100), FONT_COLOR)
        self.__draw_text("Expanded:  " + str(len(self.__expanded)), (10, self.__height-80), FONT_COLOR)
        self.__draw_text("Heuristic: " + str(self.__heuristic), (10, self.__height-60), FONT_COLOR)
        self.__draw_text("Path Cost: " + str(self.__path_cost), (10, self.__height-40), FONT_COLOR)
        self.__draw_text("Path Time: " + str(int(self.__path_time*1000)) + "ms", (10, self.__height-20), FONT_COLOR)

    # draw the grid lines
    def __draw_grid_lines(self):
        for x in range(self.__grid.width()):
            pg.draw.line(self.__screen, GRID_COLOR, (x*TILE_SIZE, 0), (x*TILE_SIZE, self.__height))
        for y in range(self.__grid.height()):
            pg.draw.line(self.__screen, GRID_COLOR, (0, y*TILE_SIZE), (self.__width, y*TILE_SIZE))

    # computes and stores a path if we have the left mouse button held down
    def __compute_path(self):
        self.__path, self.__path_cost, self.__expanded, self.__path_time, self.__heuristic = [], 0, set(), 0, 0
        if self.__m_down[0]: 
            t0 = time.clock()
            self.__path, self.__path_cost, self.__expanded = self.__grid.get_path(self.__s_tile, self.__m_tile, self.__o_size)
            self.__heuristic = self.__grid.estimate_cost(self.__s_tile, self.__m_tile)
            self.__path_time = time.clock()-t0            

    # draw a tile location with given parameters
    def __draw_tile(self, surface, tile, color, size):
        surface.fill(color, (tile[0]*TILE_SIZE, tile[1]*TILE_SIZE, TILE_SIZE*size, TILE_SIZE*size))
    
    # draws text to the screen at a given location
    def __draw_text(self, text, pos, color):
        label = self.__font.render(text, 1, color)
        self.__screen.blit(label, pos)

    # returns the tile on the grid underneath a given mouse position in pixels
    def __get_tile(self, mpos):
        return (mpos[0] // TILE_SIZE, mpos[1] // TILE_SIZE)

    # returns a pixel rectangle, given a tile on the grid
    def __get_rect(self, tile, pad):
        return (tile[0]*TILE_SIZE+pad, tile[1]*TILE_SIZE+pad, TILE_SIZE-pad, TILE_SIZE-pad)

    # called when the program is closed
    def __quit(self):
        pg.quit()                   
        sys.exit()                  

    # events and input handling
    def __events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.__quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.__quit()
                if event.key == pg.K_s:
                    self.prev_sector = -1
                    self.__o_size = 1 + (self.__o_size)%MAX_SIZE
            # reset the pathfinding start tile when the mouse button is released
            if event.type == pg.MOUSEBUTTONUP:
                self.__m_down = pg.mouse.get_pressed()
                self.__s_tile = (-1, -1)
            if event.type == pg.MOUSEBUTTONDOWN:
                # get the mouse button pressed state = [left, middle, right]
                self.__m_down = pg.mouse.get_pressed()
                # left mouse button = set start position for path finding
                if self.__m_down[0]:
                    self.__s_tile = self.__get_tile(event.pos)
                # middle mouse button = cycle through object sizes
                if self.__m_down[1]:
                    self.prev_sector = -1
                    self.__o_size = 1 + (self.__o_size)%MAX_SIZE
            # when the mouse is actiond, compute which tile it's currenlty on top of and store it
            if event.type == pg.MOUSEMOTION:
                tile = self.__get_tile(event.pos)
                self.__m_tile = (min(tile[0], self.__grid.width() - self.__o_size), min(self.__grid.height() - self.__o_size, tile[1]))

sys.setrecursionlimit(10000)

# create the game object
g = PathFindingDisplay(MAP_FILE)

# run the main game loop
while True:
    g.update()