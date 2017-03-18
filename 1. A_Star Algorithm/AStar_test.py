import sys              # used for file reading
import time             # used for timing the path-finding
from settings import *  # use a separate file for all the constant settings
import grid_solution
import grid_algo

# set the systemn recursion limit high so that we don't worry about going over
sys.setrecursionlimit(10000)

# the start and end tiles for testing
start_tiles = [(21, 3), (3, 33), (4, 50),  (2, 60), (4, 50),  (17,  0), (53, 43), (30, 33), (47,  0), (30, 34), (61, 14), (30, 34), (1, 1), (13, 8), (63, 58), (51, 23), (40, 30), (15, 32), (20, 10), (0,0)]
end_tiles   = [(46, 3), (3, 55), (13, 58), (28, 2), (13, 59), (60, 50), (30, 43), (54, 33), (60, 45), (55, 39), (10, 44), (55, 40), (5, 5), (63, 8), (63,  0), (51, 45), (20, 30), (18, 18), (40, 10), (63,63)]

# the solution grids
test_grid_solution = grid_solution.Grid(MAP_FILE)
test_grid_algo = grid_algo.Grid(MAP_FILE)



# store the correct paths and time it took to compute them
student_connected_correct = [0, 0, 0]
student_cost_correct      = [0, 0, 0]
student_path_time         = 0
solution_path_time        = 0

# compare the results of the student grid to the solution grid
print("\n------------------------------------------------------------------------------------------------------------------------------------------------")
print("|                            |  Solution Generated Values                 |   Student Generated Values                 |   Student Correct    |")
print("------------------------------------------------------------------------------------------------------------------------------------------------")
print("| TEST |   START      END    |   CONN1  CONN2  CONN3  COST1  COST2  COST3 |   CONN1  CONN2  CONN3  COST1  COST2  COST3 |   SIZE1  SIZE2  SIZE3 |")
print("------------------------------------------------------------------------------------------------------------------------------------------------")
for tile_index in range(len(start_tiles)):
    start, end = start_tiles[tile_index], end_tiles[tile_index]
    sol_cost, stu_cost = [], []
    sol_con,  stu_con  = [], []

    # compute the solution path costs and the time it takes to compute them
    t0 = time.clock()
    for size in range(1, MAX_SIZE+1):
        path_sol, cost_sol, exp_sol = test_grid_solution.get_path(start, end, size)
        sol_con.append(test_grid_solution.is_connected(start, end, size))
        sol_cost.append(cost_sol)
    time_solution = time.clock()-t0
    solution_path_time += time_solution
        
    # compute the student path costs and the time it takes to compute them
    t0 = time.clock()
    for size in range(1, MAX_SIZE+1):
        path_stu, cost_stu, exp_stu = test_grid_algo.get_path(start, end, size)
        stu_con.append(test_grid_algo.is_connected(start, end, size))
        stu_cost.append(cost_stu)
    time_student = time.clock()-t0
    student_path_time += time_student
    
    # check the student path costs for correctness
    student_correct = [True, True, True]
    for size in range(MAX_SIZE):
        if sol_con[size] == stu_con[size]:
            student_connected_correct[size] += 1
        if sol_cost[size] == stu_cost[size] and sol_con[size] == stu_con[size]: 
            student_cost_correct[size] += 1
        else:
            student_correct[size] = False
    
    print("| %4d | (%3d,%3d) (%3d,%3d) | %7r%7r%7r%7d%7d%7d | %7r%7r%7r%7d%7d%7d | %7r%7r%7r |" % (tile_index+1, start[0], start[1], end[0], end[1], 
                                                                                             sol_con[0],  sol_con[1],  sol_con[2], 
                                                                                             sol_cost[0], sol_cost[1], sol_cost[2], 
                                                                                             stu_con[0],  stu_con[1],  stu_con[2], 
                                                                                             stu_cost[0], stu_cost[1], stu_cost[2], 
                                                                                             student_correct[0], student_correct[1], student_correct[2]))

print("------------------------------------------------------------------------------------------------------------------------------------------------\n\n")
print("Student Connected  Correct:  ", student_connected_correct)
print("Student Path Costs Correct:  ", student_cost_correct, "\n")
print("Solution Path Time:          ", solution_path_time*1000, "ms")
print("Student  Path Time:          ", student_path_time*1000, "ms")