# optimization.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import itertools
import math

#import pacmanPlot
#import graphicsUtils
import util

# You may add any helper functions you would like here:
# def somethingUseful():
#     return True



def findIntersections(constraints):
    """
    Given a list of linear inequality constraints, return a list all
    intersection points.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b)
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.
    Output: A list of N-dimensional points. Each point has the form:
        (x1, x2, ..., xN).
        If none of the constraint boundaries intersect with each other, return [].

    An intersection point is an N-dimensional point that satisfies the
    strict equality of N of the input constraints.
    This method must return the intersection points for all possible
    combinations of N constraints.

    """
    # python3 autograder.py −t test_cases/q1/test_2D_1
    # python3 autograder.py −q q1
    
    # a1 * x1 + a2 * x2 + aN * xN <= b
    
    # Example Constraints: [(1,1) ,10, (-1, 0), 0, (0,-1),0]
    # Solution: [(0.0,10.0), (10.0,-0.0), (-0.0,-0.0)]
    
    #Setting up a dictionary, A matrix, B matrix, and solutions list
    constraint_dict = {}
    A_matrix = []
    B_matrix = []
    intersections = []
    
    #Retrieving the A matrix and B matrix from the constraints
    for i in range(len(constraints)):
        A, B = constraints[i]
        constraint_dict[A] = B
        tmp_list = list(A)
        A_matrix.append(tmp_list)
        B_matrix.append(B)
    
    # If B Matrix is empty then it would return an empty list
    if len(B_matrix) == 0:
        return []
    
    #Create possible combinations of the A Matrices 
    number_of_A = len(A_matrix[0])
    combinations = list(itertools.combinations(A_matrix, number_of_A))
    
    # Go through each combination pair
    for pair in combinations:
        matrix = list(pair)
        B_matrix.clear()
        
        # Find the specific B matrix for the given A combination matrix
        for i in matrix: 
            B_matrix.append(constraint_dict.get(tuple(i)))
    
        # If it is a singular matrix then continue
        if(np.linalg.det(matrix) == 0):
            continue
        # If not then solve
        else:
            x = np.linalg.solve(matrix, B_matrix)
            intersections.append(tuple(x))
    
    return intersections

    #util.raiseNotDefined()

def findFeasibleIntersections(constraints):
    """
    Given a list of linear inequality constraints, return a list all
    feasible intersection points.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b).
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.

    Output: A list of N-dimensional points. Each point has the form:
        (x1, x2, ..., xN).

        If none of the lines intersect with each other, return [].
        If none of the intersections are feasible, return [].

    You will want to take advantage of your findIntersections function.

    """
    # python3 autograder.py −t test_cases/q2/test_2D_1
    # python3 autograder.py −q q2

    # feasible if 
    # a1 * x1 + a2 * x2 + ... + aN * xN <= b for all constraints 
    
    # Find possible intersections using the function above
    intersection = findIntersections(constraints)
    
    # Set up a boolean for adding it/ not adding to the list and a list for the result
    Add = False
    feasible_intersections = []
    
    # If no possible intersections then return an empty list
    if intersection == []:
        return feasible_intersections

    # Go through each intersection
    for point in intersection:
        point = list(point)
        # Go through each constraint
        for constraint in constraints:
            # Calculate the point x constraint
            sum_1 = np.dot(point, constraint[0])
            # If it is less than equal to B, then continue 
            if sum_1 <= constraint[1]:
                Add = True
                continue
            # If just one constraint is false, break and not add to feasible_intersections
            else: 
                Add = False
                break
        
        # Append result if it fits all constraints
        if Add == True:
            feasible_intersections.append(tuple(point))
    

    return feasible_intersections

    #util.raiseNotDefined()

def solveLP(constraints, cost):
    """
    Given a list of linear inequality constraints and a cost vector,
    find a feasible point that minimizes the objective.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b).
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.

        A tuple of cost coefficients: (c1, c2, ..., cN) where
        [c1, c2, ..., cN]^T is the cost vector that helps the
        objective function as cost^T*x.

    Output: A tuple of an N-dimensional optimal point and the 
        corresponding objective value at that point.
        One N-demensional point (x1, x2, ..., xN) which yields
        minimum value for the objective function.

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    You can take advantage of your findFeasibleIntersections function.

    """
    # python3 autograder.py −t test_cases/q3/test_2D_1
    # python3 autograder.py −q q3
    
    feasible_intersections = findFeasibleIntersections(constraints)
    cost = list(cost)
    minimum = 0
    flag = 0
    point_for_optimal = None 
    
    # If the feasible intersections are empty, then just return None
    if feasible_intersections == []:
        return point_for_optimal
    
    # Iterate through each feasible intersection
    for point in feasible_intersections: 
        point = list(point)
        # Calculate the sum of objective function for each point
        sum_of_obj = np.dot(point, cost)
        
        # Only for the first one, set it equal as the minimum
        if flag == 0: 
            point_for_optimal = tuple(point)
            minimum = sum_of_obj
            flag += 1
        
        # If the sum is less than the original minimum, then it updates
        if minimum > sum_of_obj:
            point_for_optimal = tuple(point)
            minimum = sum_of_obj
            
    
    return (point_for_optimal, minimum)
    
    #util.raiseNotDefined()

def wordProblemLP():
    """
    Formulate the work problem in the write-up as a linear program.
    Use your implementation of solveLP to find the optimal point and
    objective function.

    Output: A tuple of optimal point and the corresponding objective
        value at that point.
        Specifically return:
            ((sunscreen_amount, tantrum_amount), maximal_utility)

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    """
    # python3 autograder.py −q q4
    
    # Constraint equations:
    # At least 20 fluid ounces of sunscreen (x)
    # At least 15.5 fluid ounces of tantrum (y)
    # 2.5x + 2.5y <= 100 units
    # 0.5x + 0.25y <= 50 pounds  
    # Utility (7, 4)
    
    # The constraints 
    constraints = ( ((-1, 0), -20), ((0,-1), -15.5), ((2.5, 2.5), 100), ((0.5, 0.25), 50) )
    
    # The solveLP finds the minimum but we want to find the maximum so we
    # just turn everything to utility to negative 
    utility = (-7,-4)    
    
    answer = solveLP(constraints, utility)
    answer = list(answer)
    answer[1] *= -1 
    
    return tuple(answer) 
    
    #util.raiseNotDefined()
   
def getLowerBound(constraints, index, points):
    """
    Retrieving the floor constraint 

    """
    
    # Copy the constraints into a new list 
    Constraints_left = constraints[:]
    # The floor xi <= floor(xi)
    new_con = math.floor(points[index])
    
    con_list = []
    
    for i in range(len(points)):
        if i == index:
            con_list.append(1)
        else:
            con_list.append(0)
            
    # Appending the left branch constraint (positive)
    Constraints_left.append( (tuple(con_list) , new_con) )
    
    con_list = []
    
    for i in range(len(points)):
        if i == index:
            con_list.append(-1)
        else:
            con_list.append(0)
        
    # Appending the left branch constraint (negative)
    Constraints_left.append( (tuple(con_list) , -new_con) )
    
    return Constraints_left
    
    
def getUpperBound(constraints, index, points):
    """
    Retrieving the ceil constraint

    """
    
    # Copy the constraints into the new list
    Constraints_right = constraints[:]
    # The ceiling xi >= ceil(xi)
    new_con = math.ceil(points[index])
    
    con_list = []
    
    for i in range(len(points)):
        if i == index:
            con_list.append(1)
        else:
            con_list.append(0)
    
    #Appending the right branch constraint (positive)
    Constraints_right.append( (tuple(con_list) , new_con) )
    
    con_list = []
    
    for i in range(len(points)):
        if i == index:
            con_list.append(-1)
        else:
            con_list.append(0)
    
    # Appending the right branch constraint (negative)
    Constraints_right.append( (tuple(con_list) , -new_con) )
    
    return Constraints_right

def branchAndBound(constraints, cost):
    """
    Performs Branch and Bound recursively
    """
    
    global best_sol
    
    # Find relaxed LP
    relaxedLP = solveLP(constraints, cost)
    IsInt = True
  
    # If no feasible region
    if relaxedLP == None:
        return 
    
    # Checking for all integer solutions
    for point in relaxedLP[0]:
        if (math.floor(point) - 1 * (10 ** -12)) <= point <= (math.floor(point) + 1 * (10 ** -12)):
            continue
        if (math.ceil(point) - 1 * (10 ** -12)) <= point <= (math.ceil(point) + 1 * (10 ** -12)):
            continue
        IsInt = False
    
    # If they are all integer solutions, check if the relaxedLP solution beats the best 
    if IsInt == True:
        if relaxedLP[1] < best_sol[1]:
            best_sol = relaxedLP
            return 
        else:
            return
    
    # If there is a non integer value 
    if IsInt == False:
        for i in range(len(relaxedLP[0])):
            # Check for which index has the non integer value 
            if (math.floor(relaxedLP[0][i]) - 1 * (10 ** -12)) <= relaxedLP[0][i] <= (math.floor(relaxedLP[0][i]) + 1 * (10 ** -12)):
                continue
            if (math.ceil(relaxedLP[0][i]) - 1 * (10 ** -12)) <= relaxedLP[0][i] <= (math.ceil(relaxedLP[0][i]) + 1 * (10 ** -12)):
                continue
            
            # constraint + left branch 
            new_low = getLowerBound(constraints, i, relaxedLP[0])
            # constraint + right branch
            new_high = getUpperBound(constraints, i, relaxedLP[0])
        
            # recursively try the newly added constraints 
            branchAndBound(new_low, cost)
            branchAndBound(new_high, cost)
            
    return 
    
def solveIP(constraints, cost):
    """
    Given a list of linear inequality constraints and a cost vector,
    use the branch and bound algorithm to find a feasible point with
    interger values that minimizes the objective.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b).
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.

        A tuple of cost coefficients: (c1, c2, ..., cN) where
        [c1, c2, ..., cN]^T is the cost vector that helps the
        objective function as cost^T*x.

    Output: A tuple of an N-dimensional optimal point and the 
        corresponding objective value at that point.
        One N-demensional point (x1, x2, ..., xN) which yields
        minimum value for the objective function.

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    You can take advantage of your solveLP function.

    """
    "*** YOUR CODE HERE ***"
    # python3 autograder.py −t test_cases/q5/test_2D_1
    # python3 autograder.py −q q5

    # Utilizing Global variable to keep track of the best solution
    global best_sol
    best_sol = ((), math.inf)
    
    # Do branch and bound
    branchAndBound(constraints, cost)
    
    return best_sol

    # util.raiseNotDefined()

def wordProblemIP():
    """
    Formulate the work problem in the write-up as a linear program.
    Use your implementation of solveIP to find the optimal point and
    objective function.

    Output: A tuple of optimal point and the corresponding objective
        value at that point.
        Specifically return:
        ((f_DtoG, f_DtoS, f_EtoG, f_EtoS, f_UtoG, f_UtoS), minimal_cost)

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    """
    # python3 autograder.py −q q7
    
    # The constraints
    constraints = [((1.2, 0, 0, 0, 0, 0), 30), ((0, 1.2, 0, 0, 0, 0), 30), ((0, 0, 1.3, 0, 0, 0), 30), ((0, 0, 0, 1.3, 0, 0), 30),
                   ((0, 0, 0, 0, 1.1, 0), 30), ((0, 0, 0, 0, 0, 1.1), 30), ((-1, 0, 0, 0, 0, 0), 0), ((0, -1, 0, 0, 0, 0), 0), 
                   ((0, 0, -1, 0, 0, 0), 0), ((0, 0, 0, -1, 0, 0), 0), ((0, 0, 0, 0, -1, 0), 0), ((0, 0, 0, 0, 0, -1), 0), 
                   ((-1, 0, -1, 0, -1, 0), -15), ((0, -1, 0, -1, 0, -1), -30)]
    # The cost
    cost = (12, 20, 4, 5, 2, 1)
    
    solution = solveIP(constraints, cost)
    
    if solution == None:
        return None 
    
    return solution 

    #util.raiseNotDefined()

def foodDistribution(truck_limit, W, C, T):
    """
    Given M food providers and N communities, return the integer
    number of units that each provider should send to each community
    to satisfy the constraints and minimize transportation cost.

    Input:
        truck_limit: Scalar value representing the weight limit for each truck
        W: A tuple of M values representing the weight of food per unit for each 
            provider, (w1, w2, ..., wM)
        C: A tuple of N values representing the minimal amount of food units each
            community needs, (c1, c2, ..., cN)
        T: A list of M tuples, where each tuple has N values, representing the 
            transportation cost to move each unit of food from provider m to
            community n:
            [ (t1,1, t1,2, ..., t1,n, ..., t1N),
              (t2,1, t2,2, ..., t2,n, ..., t2N),
              ...
              (tm,1, tm,2, ..., tm,n, ..., tmN),
              ...
              (tM,1, tM,2, ..., tM,n, ..., tMN) ]

    Output: A length-2 tuple of the optimal food amounts and the corresponding objective
            value at that point: (optimial_food, minimal_cost)
            The optimal food amounts should be a single (M*N)-dimensional tuple
            ordered as follows:
            (f1,1, f1,2, ..., f1,n, ..., f1N,
             f2,1, f2,2, ..., f2,n, ..., f2N,
             ...
             fm,1, fm,2, ..., fm,n, ..., fmN,
             ...
             fM,1, fM,2, ..., fM,n, ..., fMN)

            Return None if there is no feasible solution.
            You may assume that if a solution exists, it will be bounded,
            i.e. not infinity.

    You can take advantage of your solveIP function.

    """
    # python3 autograder.py −q q7
    
    M = len(W)
    N = len(C)
    
    # Set up the constraints and cost vectors 
    constraints = []    
    cost = []
    
    # xi >= 0
    for i in range(M * N):
       list_1 = [0] * (M*N)
       list_1[i] = -1
       constraints.append( (tuple(list_1), 0) )
          
    list_W = list(W)
    
    # The weight limit constraints
    for i in range(M):
        for j in range(N):
            tmp_list = [0] * (M*N)
            tmp_list[i * N + j] = list_W[i]
            constraints.append( (tuple(tmp_list), truck_limit))
    
    list_C = list(C)
    
    # The minimal food constraints 
    for i in range(M):
        tmp_list = [0] * (M*N)
        for j in range(N):
            tmp_list[j * N + i] = -1
        constraints.append( (tuple(tmp_list), -list_C[i]))
    
    # The cost of transportation
    for trancost in T:
        cost += list(trancost)
        
    solution = solveIP(constraints, tuple(cost))
    
    return solution 

    #util.raiseNotDefined()


if __name__ == "__main__":
    constraints = [((3, 2), 10),((1, -9), 8),((-3, 2), 40),((-3, -1), 20)]
    inter = findIntersections(constraints)
    print(inter)
    print()
    valid = findFeasibleIntersections(constraints)
    print(valid)
    print()
    print(solveLP(constraints, (3,5)))
    print()
    print(solveIP(constraints, (3,5)))
    print()
    print(wordProblemIP())
