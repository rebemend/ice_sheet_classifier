# Ice Shelf Compression–Extension Classification Project (Amery Ice Shelf)

## Project Scope (Updated)
This project develops an **unsupervised ice-shelf regime classifier** that labels grid points as:
- **Compression**
- **Transition**
- **Extension**

The classifier is trained using **velocity- and viscosity-based features** derived from the
**DIFFICE_jax** repository, with a **proof-of-concept focus on the Amery Ice Shelf**.

This work **does not re-train PINNs**. All physics-informed inference (velocity smoothing, strain rates,
viscosity inversion) is reused directly from the DIFFICE_jax outputs, plus an external `results.mat`
file containing anisotropic viscosities.

---

## Scientific Background

### Ice Shelf Flow Regimes
Ice shelves exhibit spatially distinct deformation regimes:

- **Compression zone**  
  Near the grounding line where longitudinal strain rate is negative.
- **Extension zone**  
  Toward the calving front where longitudinal strain rate is positive and damage accumulates.
- **Transition zone**  
  Narrow region where strain rates approach zero and rheology changes.

These regimes control:
- Buttressing strength
- Rift initiation and propagation
- Grounding-line stability

---

## Relationship to Wang et al. (2025)

The reference paper demonstrates that:

- Compression zones often follow **power-law or composite rheology**
- Extension zones violate isotropic SSA assumptions and exhibit **anisotropic viscosity**
- The transition zone marks a clear mechanical boundary

In this project, those regimes are **not imposed analytically** but instead **recovered via clustering**
in a multivariate feature space.

---

## DIFFICE_jax Repository Context

### What DIFFICE_jax Provides
From the repository:
- Structured spatial grids for Amery Ice Shelf
- Velocity fields: `u(x,y), v(x,y)`
- Ice thickness: `h(x,y)`
- Automatically differentiated strain-rate fields
- Inferred viscosities from isotropic and anisotropic SSA inversions

### External Data
- `results.mat` supplies:
  - Horizontal viscosity `μ(x,y)`
  - Vertical viscosity `η(x,y)`

This file is the **only data not already in the repository**.

---

## Problem Framing

> **Unsupervised classification of Amery Ice Shelf grid points into physical deformation regimes
using k-means clustering.**

This approach allows:
- Data-driven identification of regime boundaries
- Quantitative comparison of velocity vs. viscosity importance
- Feature ablation for physical interpretation

---

## Immediate Next Steps

1. Locate Amery Ice Shelf outputs in `DIFFICE_jax/examples/real_data`
2. Export required fields to NumPy-friendly format
3. Confirm availability of strain-rate intermediates
4. Begin feature engineering for clustering

This document provides the scientific grounding for all subsequent steps.
