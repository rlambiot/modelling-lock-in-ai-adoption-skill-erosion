# Lock-in of AI adoption through skill erosion

Code for R. Lambiotte, *Modelling lock-in of AI adoption through skill erosion* (2026), arXiv link to be added.

The code solves the model exactly at stationarity, computes the threshold set by lone holdouts, and runs the agent-based simulations. It reproduces every figure of the article and prints the numbers quoted in the text.

## The model

Each agent works either without AI (H) or with AI (A), and has an independent skill $s \in [0,1]$, the quality of the work they can produce without AI. With $x$ the fraction of AI users:

- payoffs: $\pi_H = s$ and $\pi_A = b - c\,x$, where $b$ is the benefit of AI-assisted work and $c$ the loss of distinctiveness as more people use AI;
- skills: $\dot s = \varepsilon (1 - s)$ while working without AI, and $\dot s = -\varepsilon s$ while using AI;
- behaviour: agents revise at rate $\nu$. With probability $\mu$ they choose a behaviour at random; otherwise they observe a random peer $j$ and copy them with probability $\tfrac12 + \tfrac{\beta}{2}(\pi_j - \pi_i)$.

The article uses $\mu = 0.02$ and $\beta = 0.8$, and measures time in units of $1/\nu$. The ratio of timescales is $\rho = 2\varepsilon/[\nu(1+\mu)]$, so that $\varepsilon = 0.2\nu$ gives $\rho = 0.39$.

## Contents

| File | Content |
|---|---|
| `coevo.py` | The model: exact stationary solution (`rate_coeffs`, `pdmp_stats`, `pdmp_pdf`, `exact_branch`, `branch_folds`), threshold set by lone holdouts (`holdout_threshold`), agent-based simulations (`simulate`), solution by quadrature for arbitrary switching rates and the Fermi rule (`pdmp_general`, `fermi_branch`), and imitation without noise (`hard_solve`, `hard_branch`) |
| `fig1_trajectories.py` | Fig. 1(b,c): skill of a single agent for $\rho = 0.1$ and $\rho = 3$ |
| `fig2_distributions.py` | Fig. 2: distributions of skill at the same adoption and mean skill, and the numbers of Sec. III |
| `fig3_hysteresis.py` | Fig. 3: hysteresis loop and phase diagram, and the folds of Sec. IV |
| `fig4_lockin.py` | Fig. 4: changing benefit of AI and temporary increases, and the numbers of Sec. V |
| `robustness.py` | Robustness checks of Appendix C: Fermi rule, unequal rates of learning and erosion, imitation without noise |
| `quoted_numbers.py` | Other numbers quoted in the text: bound on the imitation probability, $\theta^*$, thresholds against $\rho$, monotonicity of the stationary branch, unstable state, complementarity |
| `style.py` | Plotting style |
| `figures/` | The figures (PDF and PNG) and the cached simulation results (`*.json`) |

## Usage

```bash
pip install -r requirements.txt
python fig1_trajectories.py
python fig2_distributions.py
python fig3_hysteresis.py
python fig4_lockin.py
python robustness.py
python quoted_numbers.py
```

Run the scripts from this folder. They write the figures to `figures/` and print the numbers quoted in the article. The simulation results are cached in `figures/*.json`, so each script runs in seconds to a few minutes, mostly to recompute the exact solution. Deleting a cache file reruns the corresponding simulations, with fixed seeds; rerunning all of them takes of the order of an hour on a laptop.

The code was tested with Python 3.14, NumPy 2.3, SciPy 1.18 and Matplotlib 3.10.

## Licence

MIT, see `LICENSE`.
