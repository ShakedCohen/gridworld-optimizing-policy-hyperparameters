import numpy as np
import operator

class GridWorld:
    
    def __init__(self):
        
        self.height = 10
        self.width = 10
        self.grid = np.zeros(( self.height, self.width)) - 1

        self.wall = [[2,1],[2,2],[2,3],[2,4], [2,5], [2,6], [2,7], [2,8],
                    [3,1],
                    [4,1],[4,2],[4,3],[4,4], [4,5], [4,6], [4,7], [4,8],
                    [6,1],[6,2],[6,3],[6,4], [6,5], [6,6], [6,7], [6,8],
                    [8,1],[8,2],[8,3],[8,4], [8,5], [8,6], [8,7], [8,8],
                    ]

        # available actions
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        # random start location
        self.current_location = ( 9, np.random.randint(0,10))
        
        # set terminal states
        self.mine_location = (5,4)
        self.destination_location = (3,2)
        self.terminal_states = [self.mine_location, self.destination_location]
        
        # Set grid rewards for special cells
        self.grid[ self.mine_location[0], self.mine_location[1]] = -10
        self.grid[ self.destination_location[0], self.destination_location[1]] = 10
        
    
        
    def actions(self):
        
        return self.actions
    
    def reward(self, new_location):
        return self.grid[ new_location[0], new_location[1]]
        
    
    def step(self, action):
        
        # prev location
        last_location = self.current_location
        
        # LEFT
        elif action == 'LEFT':
            # If agent is at the left, stay still, collect reward
            if last_location[1] == 0 or [self.current_location[0], self.current_location[1] - 1] in self.wall:
                reward = self.reward(last_location)
            else:
                self.current_location = ( self.current_location[0], self.current_location[1] - 1)
                reward = self.reward(self.current_location)

        # RIGHT
        elif action == 'RIGHT':
            # If agent is at the right, stay still, collect reward
            if last_location[1] == self.width - 1 or [self.current_location[0], self.current_location[1] + 1] in self.wall:
                reward = self.reward(last_location)
            else:
                self.current_location = ( self.current_location[0], self.current_location[1] + 1)
                reward = self.reward(self.current_location)

        # UP
        if action == 'UP':
            # If agent is at the top, stay still, collect reward
            if last_location[0] == 0 or [self.current_location[0] - 1, self.current_location[1]] in self.wall:
                reward = self.reward(last_location)
            else:
                self.current_location = ( self.current_location[0] - 1, self.current_location[1])
                reward = self.reward(self.current_location)
        
        # DOWN
        elif action == 'DOWN':
            # If agent is at bottom, stay still, collect reward
            if last_location[0] == self.height - 1 or [self.current_location[0] + 1, self.current_location[1]] in self.wall:
                reward = self.reward(last_location)
            else:
                self.current_location = ( self.current_location[0] + 1, self.current_location[1])
                reward = self.reward(self.current_location)
            
        
        return reward
    
    def check_state(self):
        
        if self.current_location in self.terminal_states:
            return 'TERMINAL'