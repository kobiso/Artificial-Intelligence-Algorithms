import sys, time, random, copy
import pygame as pg
import gamestate_solution
import gamestate_student
from settings import *

# set the version of GameState we will use to display in the GUI
from gamestate_student import GameState as DisplayState

# set which version of the GameState you will use for each Player in the game
from gamestate_student import GameState as P1GameState
from gamestate_student import GameState as P2GameState

# set which Player object you will use for each Player in the game
P1Player = None
P2Player = gamestate_student.Player_AlphaBeta(0, 2000)

class Connect4:
    
    def __init__(self):
        pg.init()
        pg.display.set_caption(TITLE)
        pg.key.set_repeat(500, 100)
        self.clock  = pg.time.Clock()
        self.display_state = DisplayState(BOARD_ROWS, BOARD_COLS)
        self.width  = self.display_state.cols() * TILESIZE
        self.height = self.display_state.rows() * TILESIZE + 40
        self.screen = pg.display.set_mode((self.width, self.height))    
        self.font = pg.font.SysFont(FONTNAME, FONTSIZE, bold=FONTBOLD)
        self.winner = PLAYER_NONE
        self.text_position = (10, self.height-35)
        self.player_states = [P1GameState(BOARD_ROWS, BOARD_COLS), P2GameState(BOARD_ROWS, BOARD_COLS)]
        self.players = [P1Player, P2Player]

    # game main loop update function
    def update(self):
        self.dt = self.clock.tick(FPS) / 1000
        self.do_turn()
        self.events()
        self.draw()

    def quit(self):
        pg.quit()
        sys.exit()

    # draw the grid elements and return the surface
    def draw_board(self):
        # draw the tile rectangles
        self.screen.fill(CBLUE)
        for c in range(self.display_state.cols()):
            for r in range(self.display_state.rows()):
                self.draw_piece(self.screen, (r,c), PIECECOLOR[self.display_state.get(r,c)], 2)

    # draw a tile (r,c) location with given parameters
    def draw_piece(self, surface, tile, color, border):
        row, col = self.display_state.rows() - 1 - tile[0], tile[1]
        pg.draw.circle(self.screen, color, (col*TILESIZE+TILESIZE//2, row*TILESIZE+TILESIZE//2), TILESIZE//2-PIECEPAD)
        pg.draw.circle(self.screen, BLACK, (col*TILESIZE+TILESIZE//2, row*TILESIZE+TILESIZE//2), TILESIZE//2-PIECEPAD+border, border)
        
    # draw some text with the given arguments
    def draw_text(self, text, pos, color):
        label = self.font.render(text, 1, color)
        self.screen.blit(label, pos)

    # reset the game to a blank board
    def reset(self):
        self.winner = PLAYER_NONE
        self.display_state = DisplayState(BOARD_ROWS, BOARD_COLS)
        self.player_states[0] = P1GameState(BOARD_ROWS, BOARD_COLS)
        self.player_states[1] = P2GameState(BOARD_ROWS, BOARD_COLS)

    def do_move(self, move):
        self.display_state.do_move(move)
        self.player_states[0].do_move(move)
        self.player_states[1].do_move(move)

    # do the current turn
    def do_turn(self):
        self.winner = self.display_state.winner()
        if self.winner == PLAYER_NONE:              # there is no winner yet, so get the next move from the AI
            player = self.display_state.player_to_move()    # get the next player to move from the state
            if self.players[player] != None:        # if the current player is an AI, get its move
                self.do_move(self.players[player].get_move(self.player_states[player]))
                

    # draw everything to the screen
    def draw(self):
        self.draw_board()
        player = self.display_state.player_to_move()
        if (self.winner == PLAYER_NONE):
            self.draw_text(PLAYER_NAMES[player] + (": Human" if self.players[player] == None else ": AI Thinking"), self.text_position, PIECECOLOR[player])
        else:    
            self.draw_text(GAME_RESULT_STRING[self.winner], self.text_position, PIECECOLOR[self.winner])
        pg.display.flip()

    # returns the tile (r,c) on the grid underneath a given mouse position in pixels
    def get_tile(self, mpos):
        return (mpos[1] // TILESIZE, mpos[0] // TILESIZE)

    # returns a pixel rectangle, given a tile (r,c) on the grid
    def get_rect(self, tile, pad):
        return (tile[1]*TILESIZE+pad, tile[0]*TILESIZE+pad, TILESIZE-pad, TILESIZE-pad)

    # events and input handling
    def events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE: self.quit()
                if event.key == pg.K_r:      self.reset()
            if event.type == pg.MOUSEBUTTONDOWN:
                if pg.mouse.get_pressed()[0]:
                    move = self.get_tile(event.pos)[1]
                    if self.display_state.is_legal(move) and self.display_state.winner() == PLAYER_NONE:
                        self.do_move(move)


# A sample player that returns a random legal move
class Player_Random:

    def get_move(self, state):
        return random.choice(state.get_legal_moves())

sys.setrecursionlimit(10000)

# create the game object
g = Connect4()

# run the main game loop
while True:
    g.update()