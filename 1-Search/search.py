# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem): ## Suggestion: write a function to recurse over 
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # Commands for this search
    # python pacman.py -l tinyMaze -p SearchAgent
    # python pacman.py -l mediumMaze -p SearchAgent
    # python pacman.py -l bigMaze -z .5 -p SearchAgent
    
    # Initialize stack and visited list
    stack = util.Stack()
    visited = set()
    
    # Retrieve the start state
    start = problem.getStartState()
    # Push start state (contains tuple of (coordinate, direction), direction is an empty list)
    stack.push((start, [])) 
    # Adds the coordinate to the visited set 
    visited.add(start)
    
    # The algorithm of DFS starts, keeps looping until stack is empty
    while stack.isEmpty() == 0:
        # Current State (Top of the stack)
        coordinate, directions = stack.pop()
        
        #add the coordinate to visited
        visited.add(coordinate)
        
        # Checks if the coordinate is a goal state
        if problem.isGoalState(coordinate):
            return directions
        
        # For every successor that is not visited
        for successor in problem.getSuccessors(coordinate):
            # If the successor is visited, then continue
            if successor[0] in visited:
                continue
            # If it is not, then add it to the stack 
            else:
                stack.push((successor[0], directions + [successor[1]]))    
            
                
    return None # Return failure if it exits

    # util.raiseNotDefined()
    
    
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # Commands for this search 
    # python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
    # python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5
    
    # Initialize queue and the visited list
    queue = util.Queue()
    visited = set()
    
    # Retrieve the start state
    start = problem.getStartState()
    # Push start state (contains tuple of (coordinate, direction), direction is an empty list)
    queue.push( (start, []) ) 
    # Adds the coordinate to the visited set 
    visited.add(start)
    
    # The algorithm of BFS starts, keeps looping until queue is empty 
    while queue.isEmpty() == 0:
        coordinate, directions = queue.pop()
        
        # Checks if the coordinate is a goal state
        if problem.isGoalState(coordinate):
            return directions
        
        # For every successor that is not visited
        for successor in problem.getSuccessors(coordinate):
            # If the successor is visited, then continue
            if successor[0] in visited:
                continue
            # If it is not, then add it to visited and the queue
            else:
                visited.add(successor[0])
                queue.push( (successor[0], directions + [successor[1]]) )
                           
    return None # Return failure if it exits

    #util.raiseNotDefined()
    


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # Commands for the problem 
    # python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
    # python pacman.py -l mediumDottedMaze -p StayEastSearchAgent
    # python pacman.py -l mediumScaryMaze -p StayWestSearchAgent
    
    # Initialize priority queue and the visited list
    p_queue = util.PriorityQueue()
    visited = set()
    
    # Retrieve the start state and push onto priority queue 
    start = problem.getStartState()
    # Push start state into priority queue by Path-Cost
    p_queue.push( (start, []), 0)
    
    # The algorithm of UCS starts, keeps looping until priority queue(frontier) is empty 
    while p_queue.isEmpty() == 0:
        coordinate, directions = p_queue.pop()
        
        # checks if the coordinate is not visited
        if coordinate not in visited: 
            visited.add(coordinate)
            # Checks first if the coordinate is the goal or not
            if problem.isGoalState(coordinate):
                return directions
            # For every successor for the coordinate
            for successor in problem.getSuccessors(coordinate):
                if successor[0] not in visited:
                    p_queue.update( (successor[0], directions + [successor[1]]), problem.getCostOfActions(directions + [successor[1]]))
                
    return None # Return failure if it exits
    
    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # Commands for the problem
    # python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
    
    # Initialize priority queue and the visited list
    p_queue = util.PriorityQueue()
    visited = set()
    
    # Retrieve the start state and push onto priority queue 
    start = problem.getStartState()
    path_cost = heuristic(start, problem)
    # Push start state into priority queue by Path-Cost
    p_queue.push( (start, []) , path_cost)
    
    while p_queue.isEmpty() == 0:
        coordinate, directions = p_queue.pop()
        
        if coordinate not in visited:
            visited.add(coordinate)
            # Checks first if the coordinate is the goal or not
            if problem.isGoalState(coordinate):
                return directions
            # For every successor for the coordinate
            for successor in problem.getSuccessors(coordinate):
                if successor[0] not in visited:
                    p_queue.update( (successor[0], directions + [successor[1]]), problem.getCostOfActions(directions + [successor[1]]) + heuristic(successor[0], problem))
                    
        
    return None # Return failure if it exits
    
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
