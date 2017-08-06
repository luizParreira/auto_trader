# vim: sta:et:sw=4:ts=4:sts=4
'''
Agent module
'''
import random
import math

class Agent(object):
    '''
    Agent class responsible for choosing action and learning based upon that.
    '''
    def __init__(self, env, learning=True, random_agent=False, epsilon=1.0, alpha=0.2):
        self.env = env
        self.valid_actions = self.env.valid_actions
        self.random_agent = random_agent
        # Set the values for the learning agent
        self.q_table = dict() # Q-Table
        self.learning = learning
        self.epsilon = epsilon
        self.alpha = alpha
        self.step_data = dict()
        self.trial_data = []
        self.step = 0
        self.epsilon_decay = 0.995
        self.epsilons = []
        random.seed(1110)

    def reset(self, testing=False):
        '''
        Resets the agents' data every training and testing trial
        '''
        self.env.reset()
        self.step = 0
        if testing:
            self.epsilon = 0
            self.alpha = 0
        return

    def build_state(self):
        '''
        Builds the current state for the agent
        returns: Tuple: (,)
        '''
        state = self.env.get_current_state()

        if self.step_data.get('info'):
            holding = self.step_data['info']['currently_holding']
            state_list = list(state)
            state_list[0] = holding
            state = tuple(state_list)
        return state

    def get_max_q(self, state):
        '''
        Gets the max q value, given state
        '''
        # Calculate the maximum Q-value of all actions for a given state
        max_values = []
        max_value = 0
        max_action = ''

        # 1st find the maximum Q-Value of a given state
        for action in self.q_table[state].keys():
            value = self.q_table[state].get(action)
            if not max_action or value >= max_value:
                if not max_action or value == max_value:
                    max_values.append(action)
                else:
                    max_values = [action]
                max_value = value
                max_action = action

        return random.choice(max_values)

    def create_q(self, state):
        '''
        Creates the Q-Table for the given state, if it does not already exists.
        By populating the Q-Value over all valid actions with 0.0
        '''
        if not self.q_table.get(state):
            self.q_table[state] = dict((key, 0.0) for key in self.valid_actions)
        return

    def choose_action(self, state):
        '''
        Chooses the action for the agent:

        Random Agent:
            - Choose an action randomly

        Learning Agent:
            - Choose a random action with `epsilon` probability
            - else, choose the action with the highest Q-value for this state
        '''

        action = random.choice(self.valid_actions) # Random agent
        if self.random_agent:
            return action

        if self.learning and random.random() > self.epsilon:
            action = self.get_max_q(state)
        return action

    def learn(self, state, action, reward):
        '''
        Computes the learning that the Q-Values receive each time step.
        It recives a reward value, thats used to converge the Q-values
        to the most optimal action given a specific state.
        This function should not consider future profits, since future rewards
        are unpredictable in a trading scenario.
        '''
        # When not learning, return nothing, since the Q-table should have already been created
        if not self.learning:
            return

        # Value iteration update rule
        old_value = self.q_table[state][action]
        self.q_table[state][action] = (1.0 - self.alpha) * old_value + self.alpha * reward
        return

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
        self.epsilons.append(self.epsilon)
        self.update_epsilon()
        self.step += 1
        return

    def _collect_data(self, info, done):
        self.step_data['info'] = info
        self.step_data['done'] = done
        self.trial_data.append(self.step_data)
        return

    def update_epsilon(self):
        self.epsilon *= math.cos(0.01 * self.step)
        return
