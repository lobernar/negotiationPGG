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

if __name__ == '__main__':
    # Initialize payoff matrix
    game = negotiationPGG(len(strategies), c, r, strategies, p)
    payoffs = game.payoffs()

    # Compute estimated stationary distribution
    frequencies = []
    c0_freqs, c1_freqs, c2_freqs = [], [], []
    analytical_evolver = egt.analytical.StochDynamics(len(strategies), payoffs, Z, 2, mu)
    for beta in betas:
        sd_analytical = analytical_evolver.calculate_stationary_distribution(beta)
        c0_freqs.append(np.sum(sd_analytical[:-Z-1]))
        c1_freqs.append(np.sum(sd_analytical[-Z-1:-1]))
        c2_freqs.append(sd_analytical[-1])
    
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
    
