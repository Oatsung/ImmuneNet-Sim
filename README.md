Markdown
# ImmuneNet-Sim: A Systems Immunology Interaction Engine using Agent-Based Modeling

[![Mesa Version](https://img.shields.io/badge/Mesa-2.3.0-blue.svg)](https://github.com/projectmesa/mesa)
[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10-green.svg)]()
[![Academic Portfolio](https://img.shields.io/badge/Field-Computational%20Biology%20%7C%20AI%20in%20Healthcare-purple.svg)]()

An advanced, high-fidelity in-silico simulation platform designed to mathematically model cellular dynamics and complex chemical signaling pathways within the **Tumor Microenvironment (TME)**. Built entirely on the **Mesa 2.3.0** multi-agent architecture, the platform serves as a generic evaluation sandbox for complex, adaptive biological networks.

---

## 🔬 Core Philosophy & Cross-Disciplinary Paradigm

The conceptual architecture of `ImmuneNet-Sim` bridges the gap between **Quantitative Financial Engineering** and **Computational Systems Immunology**. 

1. **The Cross-Disciplinary Leap:** Inspired by quantitative agent-based models (ABM) used to simulate micro-structural market behaviors and liquidity shocks, this platform translates those non-linear paradigms into cellular biology. Malignant cells act as competitive agents optimizing resource capture, while immune agents operate as search-and-destroy units traversing a highly unstable spatial environment.
2. **The TME as a Hyper-Complex Sandbox:** The Tumor Microenvironment was selected as the foundational engine stress-test due to its extreme non-linear mechanics, incorporating **Stochastic Clonal Mutation**, **Chemotactic Chemokine Gradients (IFN-γ)**, and **Localized Immunosuppression (TGF-β)**.
3. **Engine Generalizability:** Engineered from the ground up to be highly generic and robust, the platform is not restricted to oncology. By recalibrating the agent decision rules and diffusion matrices, the core engine can be seamlessly adapted to simulate:
   * **Autoimmune Pathology:** Breakdown in self-tolerance and cellular self-destruction.
   * **Sepsis & Cytokine Storms:** Global hyperinflammatory cascades and systemic kinetic collapses.

---

## 📐 Mathematical & Biological Foundations

The interaction framework models cellular interactions via stochastic differential approximations overlaid onto discrete grids. 

### 1. Cellular Proliferation & Resistance Dynamics
Malignant growth is governed by a modified Lotka-Volterra constraint, subjected to targeted therapeutic mitigation and clonal evolution probabilities:

$$\frac{dC}{dt} = rC\left(1 - \frac{C}{K}\right) - \delta C T_{cell}$$

Where $C$ represents the malignant population, $r$ is the proliferation velocity, $K$ is the microenvironmental carrying capacity, and $\delta$ is the baseline T-cell cytotoxicity rate. 

During cell division, each agent possesses a stochastic mutation threshold:
* **Standard Strain ($C_{normal}$):** Subject to targeted therapy suppression ($r \times \mathbf{0.2}$).
* **Resistant Strain ($C_{mutated}$):** Probability of emergence via mutation ($P_{mut} = \mathbf{0.10}$). Bypasses targeted suppression pathways and exhibits an accelerated metabolic profile.

### 2. Paracrine Signaling & Spatial Diffusion
Two independent macro-chemical arrays control cell migration and survival across the spatial grid:
* **IFN-γ (Chemotactic Attractant Matrix):** Upregulated instantly ($+10.0$ units) at the coordinates of a successful malignant lysis event. Governs the directional vector of active immune cells.
* **TGF-β (Immunosuppressive Matrix):** Continually secreted by tumor agents ($+4.0$ by standard cells, $+8.0$ by mutated drug-resistant clones). Induces accelerated energy depletion in wild-type cytotoxic T-cells, creating a hostile biochemical fog.

Both chemical layers undergo geometric temporal degradation governed by a decay constant ($\gamma = \mathbf{0.80}$):

$$G_{t+1} = G_t \times \gamma$$

---

## ⚙️ System Architecture & Multi-Agent Design

The engine leverages a highly object-oriented paradigm utilizing a discrete 20x20 `MultiGrid` environment.

```text
       [ TissueModel (Core Engine Framework) ]
                 |                |
        +--------+--------+       +------------+
        |                 |                    |
[Spatial Grids]    [Active Agents]     [Data Collection Pipeline]
 - ifn_grid         - CancerCell        - Real-time Telemetry
 - tgf_grid         - TCell (CAR-T)     - Automated CSV Logging
                    - DirectorAgent     - Post-Simulation Plotting
🤖 Autonomous Clinical Intervention Logic
The DirectorAgent acts as an AI Oncologist—an omniscient background monitor. It intercepts system failure modes and dynamically triggers combination rescue therapies based on non-linear heuristic states:

Python
# Real-time state scanning implemented within the DirectorAgent class
should_activate = (current_orange_count >= 5) or (total_cancer >= 40) or (current_tcell_count <= 3)

if should_activate and not self.is_triggered:
    self.model.immunotherapy_active = True
    self.model.targeted_therapy_active = True
    self.model.cart_therapy_activated = True  # Supercharges T-Cells against TGF-b
📊 In-Silico Quantitative Clinical Trials
To evaluate the predictive capabilities of the architecture, high-throughput headless screening was executed across two distinct microenvironmental configurations (100 steps per trial). The emerging behaviors demonstrate excellent alignment with empirical biological phenomena:

🔴 Case A: Tumor Immune Escape (Systemic Treatment Failure)
Under high-aggression configurations, malignant proliferation outpaces baseline immune cytotoxicity. Although the DirectorAgent autonomously intercepts the crisis point at Step 29, the overwhelming accumulation of TGF-β (peaking >2000 units) triggers terminal T-Cell Exhaustion, leading to complete immune clearance and unconstrained exponential tumor expansion.

🟢 Case B: Sustained Eradication (Therapeutic Optimization)
By engineering a 2x increase in initial T-cell endurance (tcell_energy = 120), complete tumor clearance is achieved by Step 65. The engine cleanly maps the transient IFN-γ flare (chemotactic amplification during peak lysis) followed by the smooth asymptotic decay of TGF-β driven by the grid’s dissipation formulas—perfectly reproducing natural metabolic clearing post-remission.

🚀 Future Roadmap: Integrating Advanced AI
While the current engine strictly utilizes algorithmic heuristics and mathematical modeling, Phase 3 of development will transition the platform toward deep learning architectures:

Medical AI Framework Integration: Implementing frameworks like MONAI to analyze the generated spatial heatmaps as multi-channel imaging data for predictive diagnostics.

Reinforcement Learning: Upgrading the DirectorAgent from static heuristic thresholds to a Deep Q-Network (DQN), allowing it to autonomously discover optimal drug dosing schedules.

🛠 Reproducible Research & Deployment Guide
This project is optimized for Python 3.9/3.10 under a localized virtual environment.

1. Installation & Setup
Bash
git clone [https://github.com/your-username/ImmuneNet-Sim.git](https://github.com/your-username/ImmuneNet-Sim.git)
cd ImmuneNet-Sim
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
pip install mesa==2.3.0 pandas matplotlib seaborn numpy
2. Execution Pipelines
A. Interactive Advanced Medical Console UI:
Explore spatial dynamics and adjust parameters in real-time.

Bash
python app.py
# Access via [http://127.0.0.1:8521/](http://127.0.0.1:8521/)
B. Automated High-Throughput Clinical Trials:
Execute headless batch screening and generate the publication-grade 2x2 analytical matrix plot.

Bash
python run_experiments.py