# PGCA

The proposed framework integrates transfer-path physical modeling, path-level signal decomposition, adaptive graph learning, and FEV-based causal dynamic modeling for interpretable gearbox fault diagnosis and root-cause tracing.

## Highlights

- Physics-guided path-level decomposition based on transfer-path priors
- Adaptive graph learning for path interaction modeling
- FEV-based causal dynamic representation for fault propagation analysis
- Root-cause path localization with interpretable path-level outputs
- Support for ablation analysis, runtime evaluation, and compound-fault identification

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data
├── checkpoints
├── physical_modeling.m
├── PGCA.py
├── inference_pgca.py
└── interpretability_card.py
```

## Environment

Python 3.10 or above is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The code assumes a dataset layout similar to:

```text
data/
└── geardata/
    └── dec1200rpm/
        ├── G0_*.mat
        ├── G1_*.mat
        ├── G2_*.mat
        └── G3_*.mat
```

Each `.mat` file is expected to contain variables such as:

- `a_measured`: measured vibration response
- `Contribution_TD`: path-wise time-domain contributions

## Main Scripts

### 1. PGCA training and evaluation

```bash
python PGCA.py
```

This script trains the PGCA model, saves the best checkpoint, and exports:

- reconstructed predictions
- causal states `Q`
- inferred causal parameters `P`
- learned soft adjacency matrices

### 2. Inference-only evaluation

```bash
python inference_pgca.py
```

This script loads `checkpoints/best_model_all.pth` and reports:

- total inference time
- average inference time per batch
- average inference time per window

## Checkpoints

Trained checkpoints are expected under:

```text
checkpoints/
```

Examples:

- `best_model_all.pth`

## Contact

For questions regarding the code or paper, please open an issue in this repository or contact the corresponding author.
