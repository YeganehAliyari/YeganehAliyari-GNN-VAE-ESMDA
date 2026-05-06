# GNN-VAE-ESMDA

Graph neural autoencoder code for 3D mesh reconstruction, variational modeling, and ESMDA-based latent-space assimilation.

This repository contains:
- A **Python/PyTorch** pipeline for Graph AE / VAE training and evaluation on mesh point clouds.
- A **C++ GraphSampling** utility to generate pooling/unpooling connection matrices from a template mesh.

## Repository layout

```text
.
├── code/
│   ├── GraphAE_myDesign/      # Python training, inference, VAE, ESMDA, utilities
│   └── GraphSampling/         # C++ connection-matrix generator
└── README.md
```

## Main Python scripts (`code/GraphAE_myDesign`)

- `graphAE_train.py`: train the graph autoencoder (`new_AE.Model`).
- `graphAE_test.py`: evaluate a trained AE checkpoint on `.npy` point-cloud sets and export sample `.ply`.
- `Train_variational_GAE.py`: train the variational model (`Variational_GAE.VariationalAutoencoder`).
- `Train_VGAE.py`: alternate VAE training entry point.
- `GVAE_ESMDA.py`, `ESMDA_implement.py`, `ESMDA_implement_modified.py`, `Visualization.py`: latent/data assimilation and analysis workflows.
- `graphAE_param.py`: config loader (`[Record]` and `[Params]` sections), connection matrix loading, and experiment path setup.
- `graphAE_dataloader.py`: data IO/augmentation helpers (`.ply` ↔ `.npy`, batching, saving outputs).

## Main C++ utility (`code/GraphSampling`)

- `main.cpp`: loads a template `.obj`, builds multi-resolution pooling/unpooling layers, and exports connection matrices for the Python model.
- `CMakeLists.txt`: builds `GraphSampling` (requires OpenCV + zlib).

## Environment setup

Use Python 3.10+ (CUDA-enabled PyTorch recommended).

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install \
  numpy scipy pandas matplotlib opencv-python h5py scikit-learn \
  plyfile transforms3d tensorboardX trimesh \
  torch torch-geometric
```

> `torch` / `torch-geometric` versions must match your CUDA + PyTorch build.

## Expected data/config layout

This cleaned repository intentionally excludes datasets, checkpoints, and generated artifacts.  
The Python scripts expect you to provide:

1. `.npy` dataset files (e.g., train/eval/test point-cloud arrays).
2. A template mesh `.ply/.obj`.
3. Connection matrices (from `GraphSampling`) as `.npy/.npz`.
4. One or more `.config` files consumed by `graphAE_param.py`.

Most entry scripts currently use hard-coded paths near the bottom of each file.  
Before running, update those path constants to your local data/checkpoint/config locations.

## Build GraphSampling connection matrices

```bash
cd code/GraphSampling
cmake -S . -B build
cmake --build build -j
./build/GraphSampling
```

By default, `main.cpp` points to:
- input template: `../../data/Breast/Template.obj`
- output folder: `../../train/graphAE_Breast/ConnectionMatrices/`

Adjust paths in `main.cpp` as needed.

## Run typical workflows

From repository root:

```bash
cd code/GraphAE_myDesign
python graphAE_train.py
python graphAE_test.py
python Train_variational_GAE.py
python ESMDA_implement_modified.py
```

Each script is currently configured by in-file constants and/or `.config` paths; edit those first.

## Notes

- Code is GPU-first and contains `.cuda()` calls throughout.
- Several scripts are experimental/prototype style; they are executable but not yet unified behind a CLI.
- For reproducible experiments, keep your dataset paths, config files, and model checkpoints versioned outside this repo.
