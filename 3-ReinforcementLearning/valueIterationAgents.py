# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Iterate through the self.iterations
        for i in range(self.iterations):
            # Initialize a new counter 
            val = util.Counter()
            # Iterate through each of the states 
            for state in self.mdp.getStates():
                
                # If the state is non-terminal we can take the steps to finding the optimal value
                if not self.mdp.isTerminal(state):
                    # Finding a max 
                    optimalval = -9999999999
                    actions = self.mdp.getPossibleActions(state)
                    
                    # Iterate through each of the actions possible
                    for action in actions:
                        # Compute the Q-value 
                        qvalue = self.computeQValueFromValues(state, action)
                        # If the Qvalue computed is greater than equal to optimal, then we replace 
                        if qvalue >= optimalval:
                            optimalval = qvalue
                    # After iterating through the states, then add the optimal value in the counter 
                    val[state] = optimalval
            
            # Copy the first initialized counter into self.values
            self.values = val
        
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        qvalue = 0
        # Based off the equation: 
        # Q*(s,a) = sigma(s') T(s,a,s') [R(s,a,s') + discount(V*(s'))]
        # Retrieve all the possible transitions  
        transitions = self.mdp.getTransitionStatesAndProbs(state,action)
        
        # Add the qvalue according to the equation
        for i in transitions:
            qvalue += i[1] * ( self.mdp.getReward(state,action,i[0]) + self.discount * self.getValue(i[0]))
        
        return qvalue
        
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        
        # If the state is terminal, then immediately return None
        if self.mdp.isTerminal(state):
            return None
        
        # If non terminal continue 
    
        actions = self.mdp.getPossibleActions(state)
        optimalval = -9999999999
        bestaction = 0
        
        # If there are no actions, then return None as well
        if len(actions) == 0:
            return None 
        
        # Iterate through the actions
        for action in actions:
            # Compute the q value
            qvalue = self.computeQValueFromValues(state, action)
            # If computed q-value is greater than equal to the optimal value then replace action and optimalval
            if qvalue >= optimalval:
                optimalval = qvalue
                bestaction = action
        
        # Return the best action possible
        return bestaction
        
        
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Retrieve the states 
        states = self.mdp.getStates()
        
        # Iterate through self.iterations
        for i in range(self.iterations):
            
            # Since its updating only one state every iteration
            # We will say that the state is i % len(states) position
            the_state = states[i % len(states)]
            
            # Compute that states best action
            result = self.computeActionFromValues(the_state)
            
            # If there were none, then the value would be 0 
            if result == None:
                self.values[the_state] = 0
            # If there was an action found, then compute the q value for that state and action 
            else: 
                self.values[the_state] = self.computeQValueFromValues(the_state, result)
                
                    
class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        
        # Initialize a predecessor container 
        predecessors = {}
        states = self.mdp.getStates()
        
        # Iterate through the states
        for state in states:
            # If non-terminal, then process that state 
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                #Iterate through the actions found for that state 
                for action in actions:
                    # Retrieve the transition states 
                    transition_states = self.mdp.getTransitionStatesAndProbs(state,action)
                    # Iterate through the transitions
                    for t_state, prob in transition_states:
                        # if the state is in predecessors, then add it to the set
                        if t_state in predecessors:
                            predecessors[t_state].add(state)
                        # if its not in predecessors, then create a new set
                        else:
                            predecessors[t_state] = {state}
        
        # Initialize a pq 
        queue = util.PriorityQueue()
        
        # Iterate through the states
        for state in states: 
            # If non-terminal, then process that state
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                maxval = -999999999
                # To find the max value, iterate through the actions 
                for action in actions: 
                    qvalue = self.getQValue(state, action)
                    # If the qvalue computed is larger than the original max, then replace 
                    if qvalue > maxval:
                        maxval = qvalue
                # Find the absolute value of the difference 
                diff = abs(self.values[state] - maxval)
                # Update the queue, but putting in diff as a negative since its a min heap
                queue.update(state, -diff)
                 
        
        # Iterate through self.iterations
        for i in range(self.iterations):
            # If its empty exit
            if queue.isEmpty(): 
                return
            # Pop a state off from queue 
            state = queue.pop()
            
            # Update the value of state if it is non-terminal state 
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                self.values[state] = max([self.getQValue(state, action) for action in actions ])
            
            # Iterate through each predecessor of the state
            for pred in predecessors[state]:
                
                # If the state is terminal, then go on to the next state
                if self.mdp.isTerminal(pred):
                    continue
                
                maxval = -999999999
                actions = self.mdp.getPossibleActions(pred)
                
                # Iterate through each action retrieved from pred
                for action in actions: 
                    qvalue = self.getQValue(pred, action)
                    # Find the max value 
                    if qvalue >= maxval:
                        maxval = qvalue
                    
                # Find the absolute value of the difference 
                diff = abs(self.values[pred] - maxval)
                
                # If diff > theta, push the pred into the queue with priority of -dff
                if diff > self.theta:
                    queue.update(pred, -diff)
            
