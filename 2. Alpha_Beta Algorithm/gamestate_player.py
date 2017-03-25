import sys, time, random, copy
from settings import *

class GameState:

    # Initializer for the Connect4 GameState
    # Board is initialized to size width*height
    def __init__(self, rows, cols):
        self.__rows  = rows         # number of rows in the board
        self.__cols = cols          # number of columns in the board
        self.__pieces = [0]*cols    # __pieces[c] = number of pieces in a column c
        self.__player = 0           # the current player to move, 0 = Player One, 1 = Player Two
        self.__board   = [[PLAYER_NONE]*cols for r in range(rows)]

    # performs the given move, putting the piece into the appropriate column and swapping the player
    def do_move(self, move):
        if not self.is_legal(move):
            print("DOING ILLEGAL MOVE")
            sys.exit()
        self.__board[self.pieces(move)][move] = self.player_to_move()
        self.__pieces[move] += 1
        self.__player = self.opponent(self.__player)

    def undo_move(self, undomove):
        self.__pieces[undomove] -= 1
        self.__board[self.pieces(undomove)][undomove] = PLAYER_NONE
        self.__player = self.opponent(self.__player)

    # some getter functions that you probably won't need to modify
    def get(self, r, c):        return self.__board[r][c]   # piece type located at (r,c)
    def cols(self):             return self.__cols          # number of columns in board
    def rows(self):             return self.__rows          # number of rows in board
    def pieces(self, col):      return self.__pieces[col]   # number of pieces in a given column
    def total_pieces(self):     return sum(self.__pieces)   # total pieces on the board
    def player_to_move(self):   return self.__player        # the player to move next
    def opponent(self, player): return (player + 1) % 2 # return the opponent's value

    # a move (placing a piece into a given column) is legal if the column isn't full
    def is_legal(self, move):   return move >= 0 and move < self.cols() and self.__pieces[move] < self.rows()

    # returns a list of legal moves at this state (which columns aren't full yet)
    def get_legal_moves(self):  return [i for i in range(self.cols()) if self.is_legal(i)]

    # player TODO: Implement
    #
    #   Calculates a heuristic evaluation for the current GameState from the P.O.V. of the player to move
    #
    #   Args:
    #
    #     player (int) - The player whose POV the evaluation is from.
    #
    #   Returns:
    #
    #     value  (int) - A heuristic evaluation of the current GameState. Positive value should indicate
    #                    that the input player is winning, negative value that they are losing.
    #
    #     Suggested return values:
    #     Large positive value  = Player is winning the game (infinity if player has won)
    #     Larger negative value = Opponent is winning the game (-infinity if player has lost)
    #                             Infinity = Some large integer > non-win evaluations
    #
    def eval(self, player):
        winner = self.winner()
        if winner == player:
            return 1000000
        elif winner == self.opponent(player):
            return -1000000
        else:
            return self.score_check(player) - self.score_check(self.opponent(player))

    def score_check(self, player):
        score = 0
        for hori_row in range(BOARD_ROWS):
            for hori_column in range(BOARD_COLS - 3):
                poss_block = 0
                for i in range(4):
                    if self.get(hori_row, hori_column + i) != self.opponent(player): poss_block += 1
                if poss_block == 4: score += 1
        for vert_row in range(BOARD_ROWS - 3):
            for vert_column in range(BOARD_COLS):
                poss_block = 0
                for i in range(4):
                    if self.get(vert_row + i, vert_column) != self.opponent(player): poss_block += 1
                if poss_block == 4: score += 1
        for diag_up_row in range(BOARD_ROWS - 3):
            for diag_up_column in range(BOARD_COLS - 3):
                poss_block = 0
                for i in range(4):
                    if self.get(diag_up_row + i, diag_up_column + i) != self.opponent(player): poss_block += 1
                if poss_block == 4: score += 1
        for diag_down_row in range(BOARD_ROWS - 1, BOARD_ROWS - 4, -1):  # Searching from top to bottom.
            for diag_down_column in range(BOARD_COLS - 3):
                poss_block = 0
                for i in range(4):
                    if self.get(diag_down_row - 1, diag_down_column + i) != self.opponent(player): poss_block += 1
                if poss_block == 4: score += 1
        return score

    # player TODO: Implement
    # You will probably want to implement this function first and make sure it is working before anything else
    #
    #   Calculates whether or not there is a winner on the current board and returns one of the following values
    #
    #   Return PLAYER_ONE  (0) - Player One has won the game
    #   Return PLAYER_TWO  (1) - Player Two has won the game
    #   Return PLAYER_NONE (2) - There is no winner yet and the board isn't full
    #   Return DRAW        (3) - There is no winner and the board is full
    #
    #   A Player has won a connect 4 game if they have 4 pieces placed in a straight line or on a diagonal
    #   REMEMBER: The board rows and columns can be any size, make sure your checks acccount for this
    #   NOTE: Create 4 seprate loops to check win formations: horizontal, vertical, diagonal up, diagonal down
    #         Be sure to test this function extensively, if you don't detect wins correctly it will be bad
    #         Also, be sure not to check past the bounds of the board, any duplicate win checks will just
    #         end up wasting precious CPU cycles and your program will perform much worse.

    def winner(self):
        # Search the board for a horizontal win.
        for hori_row in range(BOARD_ROWS):
            for hori_column in range(BOARD_COLS - 3):
                # Grab the four horizontal pieces to be checked.
                piece_one = self.get(hori_row, hori_column)
                piece_two = self.get(hori_row, hori_column + 1)
                piece_three = self.get(hori_row, hori_column + 2)
                piece_four = self.get(hori_row, hori_column + 3)

                # Check if player one has a horizontal win.
                if ((piece_one == PLAYER_ONE) and (piece_two == PLAYER_ONE)
                    and (piece_three == PLAYER_ONE) and (piece_four == PLAYER_ONE)):
                    return PLAYER_ONE

                # Check if player two has a horizontal win.
                if ((piece_one == PLAYER_TWO) and (piece_two == PLAYER_TWO)
                    and (piece_three == PLAYER_TWO) and (piece_four == PLAYER_TWO)):
                    return PLAYER_TWO

        # Search the board for a vertical win.
        for vert_row in range(BOARD_ROWS - 3):
            for vert_column in range(BOARD_COLS):
                # Grab the four vertical pieces to be checked.
                piece_one = self.get(vert_row, vert_column)
                piece_two = self.get(vert_row + 1, vert_column)
                piece_three = self.get(vert_row + 2, vert_column)
                piece_four = self.get(vert_row + 3, vert_column)

                # Check if player one has a vertical win.
                if ((piece_one == PLAYER_ONE) and (piece_two == PLAYER_ONE)
                    and (piece_three == PLAYER_ONE) and (piece_four == PLAYER_ONE)):
                    return PLAYER_ONE

                # Check if player two has a vertical win.
                if ((piece_one == PLAYER_TWO) and (piece_two == PLAYER_TWO)
                    and (piece_three == PLAYER_TWO) and (piece_four == PLAYER_TWO)):
                    return PLAYER_TWO

        # Search the board for a diagonal up win.
        for diag_up_row in range(BOARD_ROWS - 3):
            for diag_up_column in range(BOARD_COLS - 3):
                # Grab the four diagonal up pieces to be checked.
                piece_one = self.get(diag_up_row, diag_up_column)
                piece_two = self.get(diag_up_row + 1, diag_up_column + 1)
                piece_three = self.get(diag_up_row + 2, diag_up_column + 2)
                piece_four = self.get(diag_up_row + 3, diag_up_column + 3)

                # Check if player one has a diagonal up win.
                if ((piece_one == PLAYER_ONE) and (piece_two == PLAYER_ONE)
                    and (piece_three == PLAYER_ONE) and (piece_four == PLAYER_ONE)):
                    return PLAYER_ONE

                # Check if player two has a diagonal up win.
                if ((piece_one == PLAYER_TWO) and (piece_two == PLAYER_TWO)
                    and (piece_three == PLAYER_TWO) and (piece_four == PLAYER_TWO)):
                    return PLAYER_TWO

        # Search the board for a diagonal down win.
        for diag_down_row in range(BOARD_ROWS - 1, BOARD_ROWS - 4, -1):  # Searching from top to bottom.
            for diag_down_column in range(BOARD_COLS - 3):
                # Grab the four diagonal down pieces to be checked.
                piece_one = self.get(diag_down_row, diag_down_column)
                piece_two = self.get(diag_down_row - 1, diag_down_column + 1)
                piece_three = self.get(diag_down_row - 2, diag_down_column + 2)
                piece_four = self.get(diag_down_row - 3, diag_down_column + 3)

                # Check if player one has a diagonal down win.
                if ((piece_one == PLAYER_ONE) and (piece_two == PLAYER_ONE)
                    and (piece_three == PLAYER_ONE) and (piece_four == PLAYER_ONE)):
                    return PLAYER_ONE

                # Check if player two has a diagonal down win.
                if ((piece_one == PLAYER_TWO) and (piece_two == PLAYER_TWO)
                    and (piece_three == PLAYER_TWO) and (piece_four == PLAYER_TWO)):
                    return PLAYER_TWO

        # If there are no legal moves left after checking for winners, then the game is a draw.
        if (len(self.get_legal_moves()) is 0):
            return DRAW

        return PLAYER_NONE  # The game continues!

# player TODO: Implement this class
class Player_AlphaBeta:

    # Constructor for the Player_AlphaBeta class
    #
    # Ideally, this object should be constructed once per player, and then the get_move function will be
    # called once per turn to get the move the AI should do for a given state
    #
    # Args:
    #
    #  depth      (int) - Max depth for the AB search. If 0, no limit is used for depth
    #  time_limit (int) - Time limit (in ms) for the AB search. If 0, no limit is used for time
    #
    #  NOTE: One or both of depth or time_limit must be set to a value > 0
    #        If both are > 0, then whichever happens first will terminate the AB search
    #
    def __init__(self, max_depth, time_limit):
        self.max_depth = max_depth      # set the max depth of search
        self.time_limit = time_limit    # set the time limit (in milliseconds)
        self.best_move = -1             # record the best move found so far
        self.reset()
        self.current_maxd=max_depth
        # Add more class variables here as necessary (you will probably need more)

    def reset(self):
        self.temp_best_move = -1
        self.best_move = -1
        self.best_move_val = -1000000
        self.alpha_beta_val = []

    # player TODO: Implement this function
    #
    # This function calculates the move to be perfomed by the AI at a given state
    # This function will (ideally) call your alpha_beta recursive function from the the root node
    #
    # Args:
    #
    #   state (GameState) - The current state of the Connect4 game, with the AI next to move
    #
    # Returns:
    #
    #   move (int)        - The move the AI should do at this state. The move integer corresponds to
    #                       which column to place the next piece into (0 is the left-most column)
    #
    # NOTE: Make sure to remember the current player to move, as this is the player you are calculating
    # a move for, and will act as the maximizing player throughout your AB recusive calls
    #
    def get_move(self, state):
        # reset the variables
        self.reset()
        # store the time that we started calculating this move, so we can tell how much time has passed
        self.time_start = time.clock()
        # store the player that we're deciding a move for and set it as a class variable
        self.player = state.player_to_move()
        # do your alpha beta (or ID-AB) search here
        self.ID_AB(state)
        #ab_value = self.alpha_beta(state, 0, -1000000, 1000000, True)
        # return the best move computer by alpha_beta
        return self.best_move

    def ID_AB(self, state):
        if self.max_depth == 0:
            max_d = BOARD_ROWS*BOARD_COLS
        else:
            max_d = self.max_depth
        for d in range(1, max_d + 1):
            try:
                self.current_maxd = d
                self.alpha_beta(copy.deepcopy(state), 0, -1000000, 1000000, True)
                self.best_move = self.temp_best_move
                # print(self.abval)
            except:
                break
        return self.best_move

    # player TODO: You might have a function like this... wink wink
    #
    # NOTE: Get Alpha-Beta with fixed search depth working first, then move to ID-AB. You should
    #       be able to use this alpha-beta function within your ID-AB calls.
    #
    def alpha_beta(self, state, depth, alpha, beta, max_player):

        # player TODO: Amazing recursive things that plays good
        #               See Lecture 12 notes on the course website

        # This line will determine how long has passed (in milliseconds) since you started the timer.
        # One of the most efficient and easiest ways to stop alpha-beta search after a time-out is to
        # raise an exception when you see the time passed go over the time limit. This will properly
        # unroll all of the recursive calls and exit back to the function that catches the exception.
        # If you are implementing ID-AlphaBeta, then you should have a separate function that does
        # the iterative deepening, and at each new maximum depth calls this alpha_beta function. Inside
        # that function you will catch the time out exception and stop ID-AlphaBeta, setting the best
        # move to the best move found at the last completed depth.
        #
        # NOTE: Be aware that if you just catch a default Exception, then almost ANY python error will
        # be caught by that exception and it will be difficult to debug your program. So be sure to make
        # a specific exception that you raise and catch .

        self.time_elapsed_ms = (time.clock() - self.time_start)*1000
        if self.time_limit != 0 and self.time_elapsed_ms > self.time_limit :
            raise Exception(("Timeout has occurred"))

        if depth >= self.current_maxd or state.winner() != PLAYER_NONE:
            return state.eval(self.player)

        for m in state.get_legal_moves():
            state.do_move(m)
            val = self.alpha_beta(state, depth+1, alpha, beta, False)  # insert other arguments
            state.undo_move(m)  # Must implement this method.
            if depth == 0: self.alpha_beta_val.append(val)
            if max_player and val > alpha:
                if depth == 0:
                    self.temp_best_move = m
                alpha = val
            elif not max_player and val < beta:
                beta = val
            if alpha >= beta:
                break
        return alpha if max_player else beta

        # Be aware that passing an object in python to a function does not copy that object, it
        # just passes a reference to it. Be sure to create a copy of the state to pass forward using
        # the copy.deepcopy(state) function, which is python's default way of deep-copying an object.
        # This deep copy method is the non-optimized way to implement AB, but also the easiest.
        #
        # There is an optimization that is easy to do in Connect4 which can avoid expensive state copies.
        # Instead of creating a new child state copy and then applying the move to it, you can instead
        # apply the move to the current state, pass a reference to that state into the recursive call,
        # and then undo the move after the recursive call returns.No matter how far down in the AB search
        # tree we go, every time we return a value the move will be undone and we will return back to
        # the original state after the recusion has finished. This will require the implementation of
        # an undo_move(move) function in the GameState class.

        # for now just have a placeholder that computes a random move
        # self.best_move = random.choice(state.get_legal_moves())