# vim: sta:et:sw=4:ts=4:sts=4
'''
Agent module
'''
import random

class Agent(object):
    '''
    Agent class responsible for choosing action and learning based upon that.
    '''
    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.6):
        self.env = env
        self.valid_actions = self.env.valid_actions
        # Set the values for the learning agent
        self.q_table = dict() # Q-Table
        self.learning = learning
        self.epsilon = epsilon
        self.alpha = alpha
        self.step = 0
        random.seed(1110)

    def reset(self, testing = False):
        '''
        Resets the agents' data every training and testing trial
        '''
        if testing:
            self.epsilon = 0
            self.alpha = 0

        return

    def build_state(self):
        '''
        Builds the current state for the agent
        returns: Tuple: (,)
        '''
        return

    def create_q(self, state):
        '''
        Creates the Q-Table for the given state, if it does not already exists.
        By populating the Q-Value over all valid actions with 0.0
        '''
        return self.q_table[state]

    def choose_action(self, _state):
        '''
        Chooses the action for the agent:

        Random Agent:
            - Choose an action randomly

        Learning Agent:
            - Choose a random action with `epsilon` probability
            - else, choose the action with the highest Q-value for this state
        '''
        action = random.choice(self.valid_actions)
        return action

    def learn(self, state, action, reward):
        '''
        Computes the learning that the Q-Values receive each time step.
        It recives a reward value, thats used to converge the Q-values
        to the most optimal action given a specific state.
        This function should not consider future profits, since future rewards
        are unpredictable in a trading scenario.
        '''
        # TODO: Fix this function
        return self.q_table[state][action] + reward

    def update(self):
        '''
        Function called everytime a time step is completed. It is used to compute
        all the activity that happened during this time step. It computes the sate,
        creates the state on the Q-table, if not already there, chooses an action,
        gets the reward for that action, and learns from what happened.
        '''
        state = self.build_state()
        self.create_q(state)
        action = self.choose_action(state)
        reward, _info = self.env.step(action)
        self.learn(state, action, reward)
        return
