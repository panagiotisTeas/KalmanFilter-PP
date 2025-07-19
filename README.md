# KalmanFilter‑PP

**Kalman Filtering for Particle Trajectory Reconstruction**  
*A component of my Master’s thesis*

---

## Overview

**KalmanFilter‑PP** is a Python package that implements a Kalman filter to reconstruct the trajectories of charged particles in a particle detector. It uses data from the [TrackML Particle Tracking Challenge](https://www.kaggle.com/competitions/trackml-particle-identification) and is part of a Master’s thesis exploring algorithms for tracking in high-energy physics.

---

## Motivation & Goals

The aim of this project is to explore Kalman filtering as a method for reconstructing particle paths from noisy spatial measurements in a multi-layered detector environment.

Objectives include:

- Implement a Kalman filter for particle tracking
- Test its performance on realistic, large-scale detector data
- Visualize and assess the accuracy of reconstructed tracks

---

## 🛠️ Getting Started

To get started, clone the repository and set up the environment using **Conda**:

### 1. Clone the repository

```bash
git clone https://github.com/panagiotisTeas/KalmanFilter-PP.git
cd KalmanFilter-PP
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
```

### 3. Activate the environment

```bash
conda activate kalmanfilter-pp
```

## Data

This project uses the official dataset from the TrackML Particle Tracking Challenge

Due to storage constraints, only example files may be stored locally in the data/ directory. For full datasets and competition details, please refer to the official Kaggle page above.

## Work in Progress

This project is currently under active development. Features, modules, and documentation will continue to evolve as the Kalman filter is optimized and integrated with visualization and evaluation components. Stay tuned for updates!

## License

This project is licensed under the [MIT License](LICENSE).