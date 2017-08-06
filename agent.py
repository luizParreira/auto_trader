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
        self.step_data = dict()
        self.trial_data = []
        self.step = 0
        random.seed(1110)

    def reset(self, testing=False):
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
        state = self.env.get_state()

        if self.step_data.get('info'):
            holding = self.step_data['info']['currently_holding']
            state_list = list(state)
            state_list[0] = holding
            state = tuple(state_list)
        return state

    def create_q(self, state):
        '''
        Creates the Q-Table for the given state, if it does not already exists.
        By populating the Q-Value over all valid actions with 0.0
        '''
        if not self.q_table.get(state):
            self.q_table[state] = dict((key, 0.0) for key in self.valid_actions)
        return

    def choose_action(self, _state):
        '''
        Chooses the action for the agent:

        Random Agent:
            - Choose an action randomly

        Learning Agent:
            - Choose a random action with `epsilon` probability
            - else, choose the action with the highest Q-value for this state
        '''
        action = random.choice(self.valid_actions) # Random agent
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
        return reward

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
        reward, info, done = self.env.step(action)
        self._collect_data(info, done)
        self.learn(state, action, reward)
        if done:
            print info["port_value"]
            print info["actions"]
        return

    def _collect_data(self, info, done):
        self.step_data['info'] = info
        self.step_data['done'] = done
        self.trial_data.append(self.step_data)
        return
