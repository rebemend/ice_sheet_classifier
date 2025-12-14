# Implementation Guide and Repository Structure (Updated)

## Project Focus
- **Single ice shelf**: Amery
- **Language**: Python
- **Reuse**: DIFFICE strain rates if accessible; otherwise recompute

---

## Repository Structure

```text
ice_classifier/
│
├── data/
│   ├── raw/
│   │   ├── diffice_amery/
│   │   └── results.mat
│   └── processed/
│
├── src/
│   ├── data_loading/
│   │   ├── extract_diffice_amery.py
│   │   ├── load_matlab.py
│   │   └── assemble_dataset.py
│   │
│   ├── features/
│   │   ├── strain_features.py
│   │   ├── viscosity_features.py
│   │   └── feature_sets.py
│   │
│   ├── clustering/
│   │   ├── kmeans_runner.py
│   │   ├── k_selection.py
│   │   └── ablation.py
│   │
│   ├── visualization/
│   │   ├── spatial_maps.py
│   │   └── feature_space.py
│   │
│   └── utils/
│       └── scaling.py
│
├── scripts/
│   ├── export_amery_from_diffice.py
│   ├── run_k_selection.py
│   ├── run_kmeans.py
│   └── run_ablation.py
│
├── README.md
├── requirements.txt
└── environment.yml
```

---

## Step-by-Step Workflow

### Step 1: Export DIFFICE Outputs
Script: `export_amery_from_diffice.py`
- Load Amery case from DIFFICE_jax
- Extract:
  - `x, y`
  - `u, v`
  - `h`
  - Strain-rate tensors (if available)
- Save as `.npz` or NetCDF

If strain rates are not stored, recompute via finite differences.

---

### Step 2: Merge Viscosity Data
- Load `μ`, `η` from `results.mat`
- Interpolate to DIFFICE grid if necessary
- Store unified dataset

---

### Step 3: Feature Engineering
Compute:
- `|v|`
- `∂u/∂x`
- `μ / η`

Centralize definitions in `feature_sets.py`.

---

### Step 4: K Selection
Script: `run_k_selection.py`
- Test K = 2–8
- Save inertia and silhouette plots

---

### Step 5: Final Clustering
Script: `run_kmeans.py`
- Fix K (expected: 3)
- Save labels and centroids

---

### Step 6: Visualization
- Spatial cluster map of Amery
- Feature-space scatter plots

---

### Step 7: Ablation Study (Optional)
Script: `run_ablation.py`
- Remove one feature at a time
- Compare diagnostics

---

## Deliverables
- Amery compression/transition/extension map
- K-selection diagnostics
- Physically interpretable cluster centroids

This structure is intentionally minimal, extensible, and aligned with DIFFICE_jax.

## Documentation:
- When code is finalized:
  - export conda environment
  - update requirements.txt with python dependencies and anything not available in conda
  - update README.md with instructions on how to use the repository