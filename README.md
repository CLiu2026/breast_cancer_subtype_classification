# Breast Cancer Subtype Classification from Gene Expression

## Overview
This project builds a two-stage machine learning pipeline to:
1. Distinguish healthy breast tissue from breast cancer samples.
2. Classify cancer samples into four molecular subtypes: basal, HER2-enriched, Luminal A, Luminal B.

Dataset: CuMiDa (GSE45827) - 151 samples, 54,676 gene probes.

## Repository Structure
.
|__ data/raw/ # Original CSV (not uploaded to GitHub)
|__ notebooks/ # Jupyter notebooks (full pipeline)
|__ src/ # Reusable modules
|__ results/ # Figures, reports, saved models
|__ requirements.txt
|__ README.md
## Setup
1. Clone this repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset from [CuMiDa](https://www.kaggle.com/datasets/brunogrisci/breast-cancer-gene-expression-cumida) and place `Breast_GSE45827.csv` in `data/raw/`.
4. Run the notebook `notebooks/01_full_pipeline.ipynb`.

## Results
- **Binary (Normal vs Tumor)**: Accuracy 97% (Random Forest + SMOTE)
- **Multiclass (4 subtypes)**: Accuracy 90% (Random Forest + feature selection)

## Key Figures
- PCA visualization
- Confusion matrices
- Top 20 important genes

## License
MIT
