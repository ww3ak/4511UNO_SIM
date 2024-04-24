import pandas as pd
import numpy as np
import random

import src.state_action_reward as sar


class Agent(object):
    def __init__(self, agent_info:dict):
        """Initializes the agent to get parameters and create an empty q-tables.
            q table holds values and estimates for each state-action pair
            visit table countes each visit to each state-action pair
        """

        self.epsilon     = agent_info["epsilon"]
        self.step_size   = agent_info["step_size"]
        self.states      = sar.states()
        self.actions     = sar.actions()
        self.R           = sar.rewards(self.states, self.actions)        

        self.q = pd.DataFrame(
            data    = np.zeros((len(self.states), len(self.actions))), 
            columns = self.actions, 
            index   = self.states
        )
        
        self.visit = self.q.copy()

class HumanAgent(Agent):
    '''
    This is the baseline human player that plays the first possible card availble. Each agent will play against this one 
    '''
    def __init__(self, agent_info:dict):

        super().__init__(agent_info)
        self.state_seen  = list()
        self.action_seen = list()
        self.q_seen      = list()

    def step(self, state_dict, actions_dict):
        """Choose the first playable card from the possible actions."""
        state = tuple(state_dict.values())
        
        # Identify all possible actions that are playable (i.e., action value != 0)
        actions_possible = [key for key, val in actions_dict.items() if val != 0]
        
        # Simply return the first available action
        if actions_possible:
            action = actions_possible[0]
        else:
            action = None  # No playable actions available
        
        # Track state-action pairs for updating (if needed)
        self.state_seen.append(state)
        self.action_seen.append(action)
        self.q_seen.append((state, action))
        self.visit.loc[[state], action] += 1
        
        return action
    def update(self, state_dict, action):
        pass
        
class MonteCarloAgent(Agent):

    def __init__(self, agent_info:dict):

        super().__init__(agent_info)
        self.state_seen  = list()
        self.action_seen = list()
        self.q_seen      = list()
    
    def step(self, state_dict, actions_dict):
        """
        Choose the optimal next action according to the followed policy.
        Required parameters:
            - state_dict as dict
            - actions_dict as dict
        agent decides whether to do random actiion (exploration decided by 'epsilon') 
        or action with highest estimates value (exploitaion)
        action based on e-greedy policy with probability epsilon in which a random action is chosen 
        and with probability 1-epsilon were the highsest current estimate form q table is chosen 

        specifically deals with the exploration vs exploitation using e greedy 
        random --> exploration 
        best known action accoding to Q table --> exploitation 

        e-greedy strat is designed to balance between both known rewards --> on policy 
        """
        
        # (1) Transform state dictionary into tuple
        state = [i for i in state_dict.values()]
        state = tuple(state)
        
        # (2a) Random action
        if random.random() < self.epsilon:
            
            actions_possible = [key for key,val in actions_dict.items() if val != 0]         
            action = random.choice(actions_possible)
        
        # (2b) Greedy action
        else:
            actions_possible = [key for key,val in actions_dict.items() if val != 0]
            random.shuffle(actions_possible)
            val_max = 0
            
            for i in actions_possible:
                val = self.q.loc[[state],i][0]
                if val >= val_max: 
                    val_max = val
                    action = i
        
        # (3) Add state-action pair if not seen in this simulation
        if ((state),action) not in self.q_seen:
            self.state_seen.append(state)
            self.action_seen.append(action)
        
        self.q_seen.append(((state),action))
        self.visit.loc[[state], action] += 1
        
        return action
    
    def update(self, state_dict, action):
        """
        Updating Q-values according to Belman equation
        Required parameters:
            - state_dict as dict
            - action as str

        end of an "episode" reward recieed after execution of an action state used to update Q values 
        """
        
        state  = [i for i in state_dict.values()]
        state  = tuple(state)
        reward = self.R.loc[[state], action][0]
        
        # Update Q-values of all state-action pairs visited in the simulation
        for s,a in zip(self.state_seen, self.action_seen): 
            self.q.loc[[s], a] += self.step_size * (reward - self.q.loc[[s], a])
            print (self.q.loc[[s],a])
        
        self.state_seen, self.action_seen, self.q_seen = list(), list(), list()



class ExplorationMonteCarloAgent(Agent):

    def __init__(self, agent_info:dict):

        super().__init__(agent_info)
        self.state_seen  = list()
        self.action_seen = list()
        self.q_seen      = list()

    def step(self, state_dict, actions_dict):
        """Choose a random next action from the possible actions."""
        actions_possible = [key for key, val in actions_dict.items() if val != 0]
        action = random.choice(actions_possible)  # Always select a random action
        state = tuple(state_dict.values())
        
        # Track state-action pairs for updating
        self.state_seen.append(state)
        self.action_seen.append(action)
        self.q_seen.append((state, action))
        self.visit.loc[[state], action] += 1
        
        return action
    
    def update(self, state_dict, action):
        """Update Q-values with a uniform reward assumption"""
        state = tuple(state_dict.values())
        reward = self.R.loc[[state], action][0]
        
        for s, a in zip(self.state_seen, self.action_seen):
            self.q.loc[[s], a] += self.step_size * (reward - self.q.loc[[s], a])
        
        # Clear lists after update
        self.state_seen, self.action_seen, self.q_seen = [], [], []




class ExploitationMonteCarloAgent(Agent):
    def __init__(self, agent_info:dict):

        super().__init__(agent_info)
        self.state_seen  = list()
        self.action_seen = list()
        self.q_seen      = list()

    def step(self, state_dict, actions_dict):
        """Choose the best action based on Q-values from the possible actions."""
        state = tuple(state_dict.values())

        # (2) Find the action with the maximum Q-value among possible actions
        actions_possible = [key for key, val in actions_dict.items() if val != 0]
        action = None
        val_max = float('-inf')

        for act in actions_possible:
            val = self.q.loc[[state], act][0]
            if val > val_max:
                val_max = val
                action = act

        # (3) Add state-action pair if not seen in this simulation
        if ((state), action) not in self.q_seen:
            self.state_seen.append(state)
            self.action_seen.append(action)
            self.q_seen.append(((state), action))
            self.visit.loc[[state], action] += 1

        return action
    
    def update(self, state_dict, action):
        """Update Q-values with a uniform reward assumption"""
        state = tuple(state_dict.values())
        reward = self.R.loc[[state], action][0]
        
        for s, a in zip(self.state_seen, self.action_seen):
            self.q.loc[[s], a] += self.step_size * (reward - self.q.loc[[s], a])
        
        # Clear lists after update
        self.state_seen, self.action_seen, self.q_seen = [], [], []

class SpecialCardsFirstAgent(Agent):
    def __init__(self, agent_info:dict):

        super().__init__(agent_info)
        self.state_seen  = list()
        self.action_seen = list()
        self.q_seen      = list()
    
    def step(self, state_dict, actions_dict):
        state = tuple(state_dict.values())
        actions_possible = [key for key, val in actions_dict.items() if val != 0]
        
        # Prioritize special cards
        special_cards = [a for a in actions_possible if 'skip' in a or 'reverse' in a or 'plus2' in a or 'plus4' in a]
        if special_cards and random.random() > self.epsilon:  # Use ε-greedy to sometimes explore non-special
            action = random.choice(special_cards)
        else:
            action = random.choice(actions_possible)  # Default to any possible action if no special cards

        self.state_seen.append(state)
        self.action_seen.append(action)
        self.q_seen.append((state, action))
        self.visit.loc[[state], action] += 1

        return action
    
    def update(self, state_dict, action):
        """Update Q-values with a uniform reward assumption"""
        state = tuple(state_dict.values())
        reward = self.R.loc[[state], action][0]
        
        for s, a in zip(self.state_seen, self.action_seen):
            self.q.loc[[s], a] += self.step_size * (reward - self.q.loc[[s], a])
        
        # Clear lists after update
        self.state_seen, self.action_seen, self.q_seen = [], [], []

class ColorChangeAgent(Agent):
    def __init__(self, agent_info:dict):

        super().__init__(agent_info)
        self.state_seen  = list()
        self.action_seen = list()
        self.q_seen      = list()
        
    def step(self, state_dict, actions_dict):
        """Choose action to change the color of the card in play as often as possible."""
        # (1) Transform state dictionary into tuple
        state = tuple(state_dict.values())

        # (2) Find the action (color change) with the maximum Q-value among possible actions
        actions_possible = [key for key, val in actions_dict.items() if val != 0]
        action = None
        val_max = float('-inf')

        for color in actions_possible:
            val = self.q.loc[[state], color][0]
            if val > val_max:
                val_max = val
                action = color

        # (3) Add state-action pair if not seen in this simulation
        if ((state), action) not in self.q_seen:
            self.state_seen.append(state)
            self.action_seen.append(action)
            self.q_seen.append(((state), action))
            self.visit.loc[[state], action] += 1

        return action
    
    def update(self, state_dict, action):
        """Update Q-values with a uniform reward assumption"""
        state = tuple(state_dict.values())
        reward = self.R.loc[[state], action][0]
        
        for s, a in zip(self.state_seen, self.action_seen):
            self.q.loc[[s], a] += self.step_size * (reward - self.q.loc[[s], a])
        
        # Clear lists after update
        self.state_seen, self.action_seen, self.q_seen = [], [], []



