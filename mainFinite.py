import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import egttools as egt
import seaborn as sns
import numpy as np
from myGame import C_strategy, negotiationPGG

#Initialize variables
Z = 36
#beta = 10e-5
mu = 10e-4
transitory = 1000
nb_generations = Z*10000
nb_runs = 5
betas = [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3, 10e4, 10e5]

strategies = [C_strategy(0),
              C_strategy(1),
              C_strategy(2)]
strategy_labels = ["C_"+str(i) for i, strategy in enumerate(strategies)]
c = 1
r = 1.5
p = 0.5

def fermi(beta, fitnessDiff): return 1/(1 + (np.exp(beta*fitnessDiff)))

def moran_step(current_state, beta, mu, Z, A):
    # Select 2 random players
    invader = np.random.randint(3)
    resident = np.random.randint(3)
    while resident == invader:
        resident = np.random.randint(3)
        
    # Calculate fitness
    k = current_state[invader]
    invaderFitness = (((k-1)*A[invader][invader])+((Z-k)*A[invader, resident]))/float(Z-1)
    residentFitness = ((k*A[resident][invader])+((Z-k-1)*A[resident, resident]))/float(Z-1)
    
    # Calculate imitation probabilities
    fitness_diff = invaderFitness-residentFitness
    T_plus = (1-mu)*((Z-k)/Z)*(k/(Z-1))*fermi(-beta, fitness_diff) + mu*((Z-k)/Z)
    T_minus = (1-mu)*(k/Z)*((Z-k)/(Z-1))*fermi(beta, fitness_diff) + mu*(k/Z)
        
    # Decide whether the player imitates
    rand_prob = np.random.rand()
    if rand_prob < mu:
        # Mutation occurs
        mutate_to = np.random.randint(3)
        if current_state[invader] > 0: # Check that we don't remove from empty population
            current_state[mutate_to] += 1
            current_state[invader] -= 1
    elif rand_prob < T_plus and current_state[resident] > 0:
        # Increase invaders
        current_state[invader] += 1
        current_state[resident] -= 1
    elif rand_prob < T_minus and current_state[invader] > 0: 
        # Decrease invaders
        current_state[resident] += 1
        current_state[invader] -= 1
    #print(current_state)
    return current_state


def estimate_stationary_distribution(nb_runs, transitory, nb_generations, beta, mu, Z, A):
    final_population = np.zeros(Z+1)
    c0_freq, c1_freq, c2_freq = 0.0, 0.0, 0.0
    
    # Run whole process for nb_runs
    for _ in range(nb_runs):
        # Initialize random population
        x = np.random.randint(Z+1)
        y = np.random.randint((Z+1)-x)
        initial_state = [x, y, Z-x-y] 
        current_state = initial_state
        # Transitory period
        for _ in range(transitory):
            moran_step(current_state, beta, mu, Z, A)
          
        # Run for nb_generations  
        generation_state = [0]*(Z+1)
        for _ in range(nb_generations):
            moran_step(current_state, beta, mu, Z, A)
            # Compute how often each state is reached (and average it)
            generation_state[current_state[1]] += 1/nb_generations
            
          
        final_population += generation_state
        c0_freq += current_state[0]/Z
        c1_freq += current_state[1]/Z
        c2_freq += current_state[2]/Z
    # Average the results
    final_population /= nb_runs
    c0_freq /= nb_runs
    c1_freq /= nb_runs
    c2_freq /= nb_runs
    print(c0_freq, c1_freq, c2_freq)
    
    return (c0_freq, c1_freq, c2_freq)


if __name__ == '__main__':
    # Initialize payoff matrix
    game = negotiationPGG(len(strategies), c, r, strategies, p)
    payoffs = game.payoffs()

    # Compute estimated stationary distribution
    frequencies = []
    c0_freqs, c1_freqs, c2_freqs = [], [], []
    analytical_evolver = egt.analytical.StochDynamics(len(strategies), payoffs, Z, 2, mu)
    for beta in betas:
        tmp = estimate_stationary_distribution(nb_runs, transitory, nb_generations, beta, mu, Z, payoffs)
        c0_freqs.append(tmp[0])
        c1_freqs.append(tmp[1])
        c2_freqs.append(tmp[2])
        #print(analytical_evolver.calculate_stationary_distribution(beta))
        
        
    
    
    # Plotting
    sns.set_context("notebook", rc={"lines.linewidth": 3, "axes.linewidth": 3})
    fix, ax = plt.subplots(figsize=(8, 5))
    ticks = range(len(betas))
    ax.plot(ticks, c0_freqs, label="C_0 frequency")
    ax.plot(ticks, c1_freqs, label="C_1 frequency")
    ax.plot(ticks, c2_freqs, label="C_2 frequency")
    ax.set_ylabel('stationary distribution')
    ax.set_xlabel('Intensity of selection (beta)')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='x', which='both', labelsize=15, width=3)
    ax.tick_params(axis='y', which='both', direction='in', labelsize=15, width=3)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
        
    plt.xticks(ticks, betas)

    sns.despine()
    plt.legend()
    plt.show()
    
