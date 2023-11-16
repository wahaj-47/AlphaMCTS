import numpy as np
import math
import random
from MCTSBase import TreeNode, MCTSBase, cpuct, EPS
import torch
from torch import nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, num_resBlocks, num_hidden):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 11 * 11, 121)
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 11 * 11, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
        
        
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class Node(TreeNode):
    '''
    MCT node.
    '''
    def __init__(self, state, game_end):
        # Initialize the node with the given state
        self.state = state
        self.game_end = game_end

        self.visits = np.zeros_like(state[0], dtype = float)
        self.values = np.zeros_like(state[0], dtype = float)
        self.last_action = None

        self.value_sum = 0
        self.policy = None

        self.model = ResNet(3, 64)
        self.model.eval()
    
    def is_terminal(self):
        '''
        :return: True if this node is a terminal node, False otherwise.
        '''
        return not (self.game_end == None) and 0 in self.state

    def value(self):
        '''
        :return: the value of the node form the current player's point of view
        '''
        if self.game_end == 0:
            return 0
        if self.game_end == 1:
            return -1
        else:
            x = torch.tensor(self.get_encoded_state(self.state)).unsqueeze(0)

            with torch.no_grad():
                policy, value = self.model(x)

            policy = torch.softmax(policy, axis=1).squeeze(0).reshape(11,11).cpu().numpy()
            
            policy *= np.sum(self.state, 0) == 0
            policy /= np.sum(policy)

            self.policy = policy

            return value.item()
        
    def find_action(self):
        '''
        Find the action with the highest upper confidence bound to take from the state represented by this MC tree node.
        :return: action as a tuple (x, y), i.e., putting down a piece at location x, y
        '''
        best_ucb_score = -np.inf
        best_action = None

        for index, child_visit_count in np.ndenumerate(self.visits):
            parent_visit_count = np.sum(self.visits)
            policy = self.policy[index]
            q_value = self.values[index]
            if policy != 0:
                ucb_score = q_value + cpuct * policy * math.sqrt( parent_visit_count ) / (1 + child_visit_count + EPS)
                if ucb_score > best_ucb_score:
                    best_ucb_score = ucb_score
                    best_action = index
        
        self.last_action = best_action
        return best_action

    def update(self, v):
        '''
        Update the statistics/counts and values for this node
        :param v: value backup following the action that was selected in the last call of "find_action"
        :return: None
        '''
        self.values[self.last_action] = v
        self.visits[self.last_action] += 1
        self.value_sum += v

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state[1] == 1, np.sum(state, 0) == 0, state[0] == 1)
        ).astype(np.float32)
        
        return encoded_state
    
class MCTS(MCTSBase):
    """
    Monte Carlo Tree Search
    Note the game board will be represented by a numpy array of size [2, board_size[0], board_size[1]]
    """
    def __init__(self, game):
        '''
        Your subclass's constructor must call super().__init__(game)
        :param game: the Gomoku game
        '''
        super().__init__(game)
        self.tree = {}

    def reset(self):
        '''
        Clean up the internal states and make the class ready for a new tree search
        :return: None
        '''
        self.tree = {}

    def get_visit_count(self, state):
        '''
        Obtain number of visits for each valid (state, a) pairs from this state during the search
        :param state: the state represented by this node
        :return: a board_size[0] X board_size[1] matrix of visit counts. It should have zero at locations corresponding to invalid moves at this state.
        '''
        state_str = self.game.stringRepresentation(state)
        node = self.tree.get(state_str)
        return node.visits

    def get_treenode(self, standardState):
        '''
        Find and return the node corresponding to the standardState in the search tree
        :param standardState: board state
        :return: tree node (None if the state is new, i.e., we need to expand the tree by adding a node corresponding to the state)
        '''
        state_str = self.game.stringRepresentation(standardState)
        return self.tree.get(state_str, None)

    def new_tree_node(self, standardState, game_end):
        '''
        Create a new tree node for the search
        :param standardState: board state
        :param game_end: whether game ends after last move, takes 3 values: None-> game not end; 0 -> game ends with a tie; 1-> player who made the last move win
        :return: a new tree node
        '''
        state_str = self.game.stringRepresentation(standardState)
        node = Node(standardState, game_end)
        self.tree[state_str] = node
        return node