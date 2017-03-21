import sys, time, random, copy
import gamestate_solution
import gamestate_student
from collections import OrderedDict
from settings import *

from gamestate_solution import GameState as SolState
from gamestate_student  import GameState as StuState

# a random player
class Player_Random:
    def get_move(self, state):
        return random.choice(state.get_legal_moves())

# a function for playing games
def play_game(p1, p2, players, results, total_results):
    print("\r")
    for i in range(100):print(" ", end='')
    print("\rPlaying Game: ", "{:>15s}".format(p1), " vs. ", "{:>15s}".format(p2), end='     ')
    # the solution gamestate that will keep track of the game
    state = gamestate_solution.GameState(BOARD_ROWS, BOARD_COLS)
    # the player objects to be used in this game
    player_objs = [copy.deepcopy(players[p1][0]), copy.deepcopy(players[p2][0])]
    # the states that each player will use
    player_states = [copy.deepcopy(players[p1][1]), copy.deepcopy(players[p2][1])]
    # play the game between these two players
    moves = 0
    while state.winner() == PLAYER_NONE:
        moves += 1
        # get the move for the current player
        player       = state.player_to_move()
        player_obj   = player_objs[player]
        player_state = player_states[player]
        move         = player_obj.get_move(player_state)
        # do the move in all 3 states
        print("*" if player == 0 else "* ", end='')
        sys.stdout.flush()
        state.do_move(move)
        player_states[0].do_move(move)
        player_states[1].do_move(move)        
    winner = state.winner()
    if winner == PLAYER_ONE:
        results[(p1,p2)] = (results[(p1,p2)][0] + 1, results[(p1,p2)][1], results[(p1,p2)][2])
        results[(p2,p1)] = (results[(p2,p1)][0], results[(p2,p1)][1] + 1, results[(p2,p1)][2])
        total_results[p1] = (total_results[p1][0] + 1, total_results[p1][1], total_results[p1][2])
        total_results[p2] = (total_results[p2][0], total_results[p2][1] + 1, total_results[p2][2])
    elif winner == PLAYER_TWO:
        results[(p2,p1)] = (results[(p2,p1)][0] + 1, results[(p2,p1)][1], results[(p2,p1)][2])
        results[(p1,p2)] = (results[(p1,p2)][0], results[(p1,p2)][1] + 1, results[(p1,p2)][2])
        total_results[p2] = (total_results[p2][0] + 1, total_results[p2][1], total_results[p2][2])
        total_results[p1] = (total_results[p1][0], total_results[p1][1] + 1, total_results[p1][2])
    elif winner == DRAW:
        results[(p1,p2)] = (results[(p1,p2)][0], results[(p1,p2)][1], results[(p1,p2)][2] + 1)
        results[(p2,p1)] = (results[(p2,p1)][0], results[(p2,p1)][1], results[(p2,p1)][2] + 1)
        total_results[p1] = (total_results[p1][0], total_results[p1][1], total_results[p1][2] + 1)
        total_results[p2] = (total_results[p2][0], total_results[p2][1], total_results[p2][2] + 1)

# define players that will play in a round-robin tournament and the states they will use
players = {
    "Random"     : (Player_Random(), SolState(BOARD_ROWS, BOARD_COLS)),
    "SOL-AB-D1"  : (gamestate_solution.Player_AlphaBeta(1, 0), SolState(BOARD_ROWS, BOARD_COLS)),
    "SOL-AB-D2"  : (gamestate_solution.Player_AlphaBeta(2, 0), SolState(BOARD_ROWS, BOARD_COLS)),
    "SOL-AB-D3"  : (gamestate_solution.Player_AlphaBeta(3, 0), SolState(BOARD_ROWS, BOARD_COLS)),
    #"SOL-AB-T1"  : (gamestate_solution.Player_AlphaBeta(1, 1000), SolState(BOARD_ROWS, BOARD_COLS)),
    "STU-AB-D1"  : (gamestate_student.Player_AlphaBeta(1, 0), StuState(BOARD_ROWS, BOARD_COLS)),
    "STU-AB-D2"  : (gamestate_student.Player_AlphaBeta(2, 0), StuState(BOARD_ROWS, BOARD_COLS))
}

# order the dictionary by player name
players = OrderedDict(sorted(players.items(), key=lambda t: t[0]))
             
# dictionary of game results
# results(pname1, pname2) = (wins, losses, draws) for pname1
results, total_results = {}, {}
for name1, player1 in players.items():
    total_results[name1] = (0, 0, 0)
    for name2, player2 in players.items():
        if not (name1, name2) in results:
            results[(name1, name2)] = (0, 0, 0)

# play a number of games per pairing
games_per_pair = 4
for name1, player1 in players.items():
    for name2, player2 in players.items():
        if name1 == name2: continue
        for g in range(games_per_pair):
            play_game(name1, name2, players, results, total_results)

# print the results of the games to the screen
# results will be printed in alphabetical order by name
fmt_s = "{:>15s}"
fmt_r = "   {:>3d}W{:>3d}L{:>3d}D"
print("\n\n")
print(fmt_s.format(""), end='')
for name1, player1 in players.items():
    print(fmt_s.format(name1), end='')
print(fmt_s.format("Total"))
for name1, player1 in players.items():
    print(fmt_s.format(name1), end='')
    for name2, player2 in players.items():
        if name1 == name2: print(fmt_s.format("-"), end='')
        else: print(fmt_r.format(results[(name1, name2)][0], results[(name1, name2)][1],results[(name1, name2)][2]), end='')
    print(fmt_r.format(total_results[name1][0], total_results[name1][1], total_results[name1][2]))
