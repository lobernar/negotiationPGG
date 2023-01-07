from typing import Union, List
import numpy as np
import egttools as egt
from egttools.games import AbstractTwoPLayerGame, AbstractNPlayerGame
from egttools import sample_simplex, calculate_nb_states

C = 1
D = 0

class C_strategy():
    def __init__(self, k : int):
        self.k = k
        
    def get_action(self, curr_act, state):
        nb_cooperators = 0
        for action in state:
            if action == C: nb_cooperators += 1
        # Cooperate only if at least k other players cooperate
        if nb_cooperators-curr_act >= self.k or self.k == 0: return C
        else: return D
    
    @property
    def type(self) -> str:
        return "C" + str(self.k)

class negotiationPGG(AbstractTwoPLayerGame):
    def __init__(self, nb_strategies: int, cost, multiplying_factor, strategies : List[C_strategy], p : float):
        self.nb_group_configurations_ = 6
        self.group_size_ = 2
        self.c = cost
        self.r = multiplying_factor
        self.p = p
        self.strategies = strategies
        self.state = self.create_init_state() # Creates a random initial state
        super().__init__(nb_strategies) # This constructor calls calculate_payoffs()
        
        # hardcoding the payoffs yields the correct results
        x = -self.c + self.r*self.c
        self.payoffs_ = np.array([
            [x, x, -self.c+(self.r*self.c)/2],
            [x, self.p*x, 0],
            [(self.r*self.c)/2, 0, 0]
        ])
                
    def create_init_state(self):
        # With probability p players initially choose to cooperate and (1-p) to defect
        state = []
        for _ in range(self.group_size_):
            if (np.random.rand() < self.p): state.append(C)
            else: state.append(D)
        return state
    
    def play(self, group_composition: Union[List[int], np.ndarray], game_payoffs: np.ndarray) -> None:
        self.state = self.create_init_state()
        non_zero = []
        # Transform group composition to [x, y] meaning player 1 uses strategy x and player 2 strategy y
        players = []
        for i, strategy_count in enumerate(group_composition):
            if strategy_count != 0: non_zero.append(i)
            for _ in range(strategy_count):
                players.append(i)
                
        self.negotiate(players)
        contributions = 0.0
        for i in range(len(players)):
            action = self.state[i] # Choose action according to negotioation process
            if action == 1:
                #contributions += strategy_count*self.c
                contributions += self.c
                game_payoffs[players[i]] = - self.c
        
        benefit = (contributions*self.r) / self.group_size_
        game_payoffs[non_zero] += benefit
    
    def calculate_payoffs(self) -> np.ndarray:
        payoffs_container = np.zeros(shape=(self.nb_strategies_,), dtype=np.float64)
        for i in range(self.nb_group_configurations_):
            # Get group composition
            group_composition = sample_simplex(i, self.group_size_, self.nb_strategies_)
            self.play(group_composition, payoffs_container)
            players = []
            for j, strategy_count in enumerate(group_composition):
                for _ in range(strategy_count):
                    players.append(j)
            self.payoffs_[players[0], players[1]] += payoffs_container[players[0]]
            if(players[0] != players[1]): self.payoffs_[players[1], players[0]] += payoffs_container[players[1]]
                    
            # Reinitialize payoff vector
            payoffs_container[:] = 0

        return self.payoffs()
    
    
    def negotiate(self, players):
        stationary_state = False
        prev_state = self.state[:]
        
        # Repeat until stationary state is reached (such a state always exists)
        while not stationary_state:
            # Select random player
            player_index = np.random.randint(len(players))
            
            # Change thought according to strategy
            action = self.strategies[players[player_index]].get_action(self.state[player_index], self.state)
            self.state[player_index] = action 
                       
            # Check if stationary state is reached
            if prev_state == self.state: 
                # Check if any player wished to change their thought
                stationary_state = True
                for i in range(len(players)):
                    action = self.strategies[players[i]].get_action(self.state[i], self.state)
                    if(action != self.state[i]):
                        self.state[i] = action 
                        stationary_state = False
                        
            prev_state = self.state[:]