# SIT724 — Multi-Robot Intersection Management

This repository contains the simulation environment and controller implementations developed for the SIT724 Honours Thesis project at Deakin University.

The project compares multiple autonomous intersection-management approaches within a shared four-way multi-robot traffic environment.

## Implemented Controllers

* Strict FCFS Controller
* Colour-Priority Reservation Controller
* Parallel-Compatible FCFS (PC-FCFS) Controller
* Matrix-Based Intersection Manager
* Learning-Based Neural-Network Controller

## Repository Files

```text
fcfs.py              # Strict FCFS controller
cbrs.py              # Colour-priority reservation controller
pc_fcfs.py           # Parallel-compatible FCFS controller
m_im.py              # Matrix-based intersection manager
nn_im.py             # Neural-network intersection manager
train_nn_model.py    # Neural-network training pipeline
```

## Features

* Four-way intersection simulation
* Multi-robot traffic coordination
* Rule-based and learning-based controllers
* Traffic metric logging
* Neural-network training and inference
* Matplotlib-based visualisation

## Neural-Network Configuration

* PyTorch implementation
* Fully connected feedforward network
* Hidden layers: 256 → 128 → 64
* ReLU activations
* Batch normalisation
* Dropout regularisation
* Adam optimiser
* BCEWithLogitsLoss objective

## Requirements

```bash
pip install numpy matplotlib torch scikit-learn joblib
```

## Running the Controllers

```bash
python fcfs.py
python cbrs.py
python pc_fcfs.py
python m_im.py
python train_nn_model.py
python nn_im.py
```

## Research Context

This repository accompanies the Honours Thesis:

**“Comparative Study of Rule-Based, Hybrid, and Learning-Based Intersection Management for Multi-Robot Systems”**

Deakin University
Bachelor of Software Engineering (Honours)

## Author

Vidul Attri
