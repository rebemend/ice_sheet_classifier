# K-Means Classification Plan (Amery Ice Shelf)

## Why K-Means Is Appropriate

### Problem Properties
- No labeled ground truth
- Physically expected small number of regimes
- Continuous, dense spatial data
- Emphasis on interpretability

**K-means** is appropriate because:
- It is unsupervised
- Centroids are physically interpretable
- It scales to large ice-shelf grids
- Diagnostics (inertia, silhouette) are simple and transparent

---

## Hypothesis on Number of Clusters

### Primary Hypothesis
**K = 3** clusters corresponding to:
1. Compression
2. Transition
3. Extension

Expect an elbow in the inertia vs k plot at k = 3. 
Expect silhouette to be above ~0.3 but ideally above 1.

### Validation
- Elbow method (inertia vs. K)
- Silhouette score
- Spatial coherence on the Amery shelf
- Consistency with longitudinal strain-rate sign

---

## Feature Sets

### Core Deformation Features (Baseline)
| Feature | Description |
|------|-------------|
| `∂u/∂x` | Longitudinal strain rate |
| `|v|` | Velocity magnitude |

Purpose: test if deformation alone recovers regimes.

---

### Velocity + Viscosity (Primary Model)
| Feature | Description |
|------|-------------|
| `∂u/∂x` | Longitudinal strain |
| `|v|` | Speed |
| `μ` | Horizontal viscosity |
| `μ/η` | Anisotropy ratio |

This is the **main production configuration**.

---

### Extended Physics (Diagnostic)
- Ice thickness `h`
- Velocity gradients
- Spatial coordinates `(x,y)` (used cautiously)

---

## Scaling and Preprocessing
All features:
- Flattened per grid point
- Standardized (zero mean, unit variance)

---

## Expected Outputs

### Feature-Space Plots
- Strain vs. anisotropy
- Speed vs. viscosity
- Colored by cluster

### Spatial Maps
- Amery Ice Shelf colored by cluster
- Overlaid velocity magnitude or grounding line

---

## Feature Ablation (Optional)
Procedure:
1. Run baseline clustering
2. Remove one feature
3. Compare inertia and silhouette

Interpretation:
- Large degradation ⇒ feature is physically important

---

## Known Limitations
- K-means assumes compact clusters
- Transition zones may be diffuse
- No spatial smoothness constraint

These are acceptable for a proof of concept.
