# Major-Occupation Specificity and College Skill Production

This repository contains the code used for the analysis in the paper "Major-Occupation Specificity and College Skill Production" by Li, Linde, and Shimao. The paper was formerly circulated as "Major Complexity Index and College Skill Production ".

## Data Requirement
The analysis in this project uses data from two sources:
- **American Community Survey (ACS)**
- **National Survey of Student Engagement (NSSE)**


## Instructions

### 1. Prepare ACS Data
First, run the `ACS_data_prep.py` script to process the ACS data. The processed data will be stored in the `data/processed_data` directory.

### 2. Compute Major-Level Indices
Next, use the following scripts to compute the major-level indices:
- `compute_MOS.py`: Computes the Major-Occupation Specificity (MOS) index.
- `compute_other_index.py`: Computes additional indices for the analysis.

### 3. Replicate Main Results
To replicate the results from the main article, run the following scripts:
- `main_regressions.py`: Runs the primary regressions used in the paper.
- `main_extra_analyses.py`: Performs additional analyses.

Additionally, you can format the results into LaTeX tables by running the `gen_latex_tables.py` script.

### 4. Generate Appendix Simulations
The following Jupyter notebooks generate graphs used in the Appendix:
- `appendix_simulations1_data_requirements.ipynb`: Simulates and visualizes data requirements.
- `appendix_simulations2_hyperparameters.ipynb`: Simulates and visualizes the impact of hyperparameter choices.

You can open and run these notebooks in any Jupyter environment.
