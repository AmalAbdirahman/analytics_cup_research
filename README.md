# SkillCorner X PySport Analytics Cup
This repository contains the submission template for the SkillCorner X PySport Analytics Cup **Research Track**. 
Your submission for the **Research Track** should be on the `main` branch of your own fork of this repository.

Find the Analytics Cup [**dataset**](https://github.com/SkillCorner/opendata/tree/master/data) and [**tutorials**](https://github.com/SkillCorner/opendata/tree/master/resources) on the [**SkillCorner Open Data Repository**](https://github.com/SkillCorner/opendata).

## Submitting
Make sure your `main` branch contains:
1. A single Jupyter Notebook in the root of this repository called `submission.ipynb`
    - This Juypter Notebook can not contain more than 2000 words.
    - All other code should also be contained in this repository, but should be imported into the notebook from the `src` folder.
2. An abstract of maximum 500 words that follows the **Research Track Abstract Template**.
    - The abstract can contain a maximum of 2 figures, 2 tables or 1 figure and 1 table.
3. Submit your GitHub repository on the [Analytics Cup Pretalx page](https://pretalx.pysport.org)

Finally:
- Make sure your GitHub repository does **not** contain big data files. The tracking data should be loaded directly from the [Analytics Cup Data GitHub Repository](https://github.com/SkillCorner/opendata).For more information on how to load the data directly from GitHub please see this [Jupyter Notebook](https://github.com/SkillCorner/opendata/blob/master/resources/getting-started-skc-tracking-kloppy.ipynb).
- Make sure the `submission.ipynb` notebook runs on a clean environment.

_⚠️ Not adhering to these submission rules and the [**Analytics Cup Rules**](https://pysport.org/analytics-cup/rules) may result in a point deduction or disqualification._

---

# Schrödinger's Pitch Control: Quantifying the Decay of Truth
## SkillCorner X PySport Analytics Cup (Research Track)

**Match ID:** 1886347

---

### Abstract

**Introduction**

Broadcast tracking data suffers from a critical limitation: occlusion. When players leave the camera view, standard models use linear extrapolation (`v_last`), introducing "Deterministic Bias." In high-stakes tactical analysis, assuming a player is at a specific coordinate when they are actually unknown creates a false sense of security. This study proposes a probabilistic framework to quantify this uncertainty.

**Methods**

We discard the single-point coordinate system for occluded players ("Ghosts"). Instead, we model their position as a Probability Density Function (PDF) using **Anisotropic Particle Filtering**. The physics engine applies Langevin dynamics, where stochastic noise is biased by the player's momentum vector. We quantify the tactical impact by computing a **Probabilistic Pitch Control** surface and subtracting the deterministic baseline to isolate the "Zone of Deception."

**Results**

Applied to Match 1886347, our model reveals significant divergence between broadcast data and tactical reality. We identified that **60% of high-velocity sprints (>4m/s)** in wide channels are invisible to the camera. Furthermore, in transition phases, the probabilistic model identified a **+32.4 Unit** "Ghost Gain"—defensive control invisible to standard models.

**Conclusion**

We demonstrate that uncertainty is signal, not noise. By quantifying the "Decay of Truth," we provide analysts with a confidence interval for their tactical models.
