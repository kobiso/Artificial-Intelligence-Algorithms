# define some colors (R, G, B)
WHITE       = (255, 255, 255)
BLACK       = (  0,   0,   0)
DARK_GREY   = ( 40,  40,  40)
LIGHT_GREY  = (100, 100, 100)
GREEN       = (  0, 255,   0)
BLUE        = (  0,   0, 255)
RED         = (255,   0,   0)
YELLOW      = (255, 255,   0)
PURPLE      = (255,   0, 255)
TILE_COLOR  = [LIGHT_GREY, GREEN, BLUE]

# display settings
FPS         = 60
TITLE       = "GridWorld Pathfinding Visualization"
FONT_NAME   = "monospace"
FONT_SIZE   = 16
FONT_BOLD   = True
FONT_COLOR  = WHITE
BG_COLOR    = LIGHT_GREY
GRID_COLOR  = DARK_GREY
TILE_SIZE   = 12
MAP_FILE    = 'map.txt'
MAX_SIZE    = 3
DRAW_EXPANDED = True

# pathfinding settings
DIAGONAL_COST = 141
CARDINAL_COST = 100
LEGAL_ACTIONS = [(-1, -1), (0, -1), (1, -1),
                 (-1,  0),          (1,  0),
                 (-1,  1), (0,  1), (1,  1)]
