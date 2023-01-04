import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import egttools as egt
from egttools.plotting.simplified import plot_replicator_dynamics_in_simplex
from myGame import negotiationPGG, C_strategy

strategies = [C_strategy(0),
              C_strategy(1),
              C_strategy(2)]
strategy_labels = ["C_"+str(i) for i, strategy in enumerate(strategies)]
c, r, p = 1, 1.6, 0

game = negotiationPGG(len(strategies), c, r, strategies, p)
print(game.payoffs())

fig, ax = plt.subplots(figsize=(10,8))

simplex, gradients, roots, roots_xy, stability = plot_replicator_dynamics_in_simplex(game.payoffs(), ax=ax)

plot = (simplex.draw_triangle()
           .draw_gradients(density=1)
           .add_colorbar(label='gradient of selection')
           .add_vertex_labels(strategy_labels, epsilon_bottom=0.12)
           .draw_stationary_points(roots_xy, stability)
           .draw_scatter_shadow(lambda u, t: egt.analytical.replicator_equation(u, game.payoffs()), 100, color='gray', marker='.', s=0.1)
          )

ax.axis('off')
ax.set_aspect('equal')

plt.xlim((-.05,1.05))
plt.ylim((-.02, simplex.top_corner + 0.05))
plt.show()
