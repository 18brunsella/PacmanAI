# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # Return the qvalue set in the self.values counter
        qvalue = self.values[(state, action)]
        return qvalue
        
        #util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        maxval = -9999999999
        # Retreive the actions for the state
        actions = self.getLegalActions(state)
        
        # If there are no actions, return 0 
        if len(actions) != 0:
            # Iterate through the actions
            for action in actions:
                # Compute the q values for each action
                qvalue = self.getQValue(state, action)
                # If it is greater than maxval then replace it 
                if qvalue > maxval: 
                    maxval = qvalue
            return maxval
        else: 
            return 0 
        
        #util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        maxval = -9999999999
        # Retrieve the actions for the state
        actions = self.getLegalActions(state)
        bestaction = None 
        
        # If there are no actions, return None
        if len(actions) != 0:
            # Iterate through each action
            for action in actions:
                # Compute the q values for each action
                qvalue = self.getQValue(state, action)
                # If it is greater than the current max, replace bestaction and maxval
                if qvalue > maxval:
                    bestaction = action 
                    maxval = qvalue
                # if its equal then use random.choice to choose one 
                elif qvalue == maxval:
                    bestaction = random.choice([bestaction, action])
            return bestaction
        else:
            return bestaction
        
        #util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        
        # If there are no actions, return the None 
        if len(legalActions) != 0:
            # Flip the coin 
            bool_res = util.flipCoin(self.epsilon)
            # If its true then use random.choice 
            if bool_res:
                action = random.choice(legalActions)
            # If its false, then compute for the action from q value 
            else:
                action = self.computeActionFromQValues(state)
        
        return action
    
        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # Retrieve the q value for the state and action
        qvalue = self.getQValue(state, action)
        # Update the new q value 
        # (1 - alpha) * qvalue + alpha * (reward + discount * qvalue)
        self.values[(state, action)] = (1- self.alpha) * qvalue + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))
         
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        qvalue = 0
        # Retrieve features
        featureVector = self.featExtractor.getFeatures(state,action)
        print(featureVector)
        # Compute the qvalue 
        # Sum up self.weights * featureVector for each features 
        for i in featureVector:
            qvalue += self.weights[i] * featureVector[i]
        
        return qvalue
        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # Retrieve features
        featureVector = self.featExtractor.getFeatures(state,action)
        print(featureVector)
        # Calculate the difference:
        # reward + (discount * qvalue(nextstate)) - qvalue(currentstate)
        difference = (reward + (self.discount * self.getValue(nextState))) - self.getQValue(state,action)
        # Update the counter for self.weights[feature]
        for i in featureVector:
            self.weights[i] = self.weights[i] + self.alpha * difference * featureVector[i]
        
        #util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            
            pass
