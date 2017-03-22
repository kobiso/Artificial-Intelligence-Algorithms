# define some colors (R, G, B)
BLACK       = (  0,   0,   0)
WHITE       = (255, 255, 255)
RED         = (255,   0,   0)
YELLOW      = (255, 255,   0)
CBLUE       = (  0, 111, 185)
PIECECOLOR  = [YELLOW, RED, WHITE, WHITE]

# game settings
FPS       = 60
TITLE     = "Connect 4"
FONTNAME  = "monospace"
FONTSIZE  = 30
FONTBOLD  = True
TILESIZE  = 110
PIECEPAD  = 5
DRAW_EXPANDED = True

# connect settings
PLAYER_ONE  = 0
PLAYER_TWO  = 1
PLAYER_NONE = 2
DRAW        = 3
BOARD_ROWS  = 6
BOARD_COLS  = 7

# result strings
PLAYER_NAMES = ["Player One", "Player Two", "Player None", "Draw"]
GAME_RESULT_STRING = ["Player One Wins!", "Player Two Wins!", "Game In Progress", "Game is a Draw!"]