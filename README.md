# Exo-Anomaly-Pipeline: 3D Cosmic Habitability & Anomaly Detection Engine
An advanced computational astrophysics and predictive machine learning pipeline designed to ingest, process, and isolate thermodynamic and geometric anomalies across thousands of exoplanets using live data from the NASA Exoplanet Archive.

Developed as an independent passion project for university application profiling (VU Amsterdam).

---

## 🔬 Core Architecture & System Logic

This engine operates as a hybrid AI system combining vectorized physics data engineering with supervised and unsupervised machine learning models.

1. **Vectorized In-Memory Data Engineering (DuckDB)**: Pulls live astronomical records via `astroquery` and executes high-speed, vectorized SQL arrays to compute planetary bulk density indexes, Einstein radius lensing proxies, and stellar flux distributions.
2. **Kopparapu Thermodynamics**: Implements multi-layer habitable zone boundary thresholds based on effective stellar flux scaling and equilibrium blackbody temperature equations ($A_B = 0.3$).
3. **Synthetic Minority Over-sampling (SMOTE)**: Dynamically handles extreme class imbalances (since habitable candidates represent $<2\%$ of the galaxy) by scaling nearest-neighbors ($k$) relative to data density to prevent data leakage and system crashes.
4. **Automated Hyperparameter Optimization (Optuna)**: Executes Bayesian optimization pathways to tune the objective loss function of a `LightGBM` Binary Classifier.
5. **Unsupervised Anomaly Isolation (Isolation Forest)**: Isolates structural anomalies and physical outliers (such as ultra-dense sub-stellar objects) without prior target labels.

---

## 📊 Live Discoveries & System Outputs

The engine successfully separates standard galactic profiles from high-energy furnaces and high-probability water worlds.

### Standard Profiles (Circles)
* **Kepler-1373 b**: A standard close-in hot Sub-Neptune ($T_{eq} \approx 2,441\text{ K}$). Correctly rejected by the LightGBM classifier (`Habitability_Index: 5.57%`) and labeled as a Standard Profile due to its predictable galactic distribution.

### Anomalous Profiles (Squares / Diamonds)
* **LHS 1140 b**: Flagged as a massive anomaly ($0.648$) due to extreme bulk density. Successfully identified as an elite habitable candidate (`Habitability_Index: 94.43%`) with an equilibrium temperature of $201.43\text{ K}$.
* **HATS-70 b**: Triggered an extreme anomaly response ($0.667$). Despite being non-habitable, its massive scale ($\approx 4,100\text{ Earth Masses}$ / 13 Jupiter Masses) packed into a tight $0.036\text{ AU}$ orbit caused the Isolation Forest to isolate it from standard planetary arrays.

---

## 🪐 5D Dark Cosmic Visualization Map
The pipeline projects all output vectors onto an interactive 3D space using Plotly, utilizing the Cividis sequential scale. 
* **Color Scale**: Yellow/Orange = High Habitability | Dark Blue = Non-Habitable.
* **Shape/Size Scale**: Large Diamonds = Anomalous Profiles | Small Circles = Standard Profiles.

![3D Graph Map](your_image_filename.png)

---

## 🛠️ Dependencies & Execution
To deploy the pipeline, install the complete mathematical and astronomical stack:
```bash
pip install optuna lightgbm duckdb astroquery plotly scikit-learn pandas numpy imbalanced-learn
```
