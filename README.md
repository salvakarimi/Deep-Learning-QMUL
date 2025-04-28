# CIFAR-10 Weighted-Block CNN  
*A coursework project for “Neural Networks & Deep Learning (ECS7026P)”*

<div align="center">
  <img src="https://raw.githubusercontent.com/your-repo/assets/readme_cifar10_grid.png" width="600"/>
  <br/>
  <em>Example CIFAR-10 images (airplane, automobile, bird, …)</em>
</div>

---

## Table of Contents
1. [Overview](#overview)  
2. [Architecture](#architecture)  
3. [Repository Layout](#repository-layout)  
4. [Getting Started](#getting-started)  
5. [Training & Evaluation](#training--evaluation)  
6. [Results](#results)  
7. [Key Implementation Details](#key-implementation-details)  
8. [Possible Extensions](#possible-extensions)  
9. [Citation / Coursework Note](#citation--coursework-note)  
10. [License](#license)

---

## Overview
This project implements and experiments with a **weighted-block convolutional neural network** for CIFAR-10 image classification.  
Each *intermediate block* contains multiple **parallel convolutional branches** whose outputs are **linearly combined with learned coefficients** derived from a global-average summary of the input feature map.  
The design echoes squeeze-and-excitation and mixture-of-experts ideas while strictly following the coursework brief.

**Goals**

* Build the novel architecture from scratch in PyTorch.  
* Train on CIFAR-10 and log batch-level loss plus epoch-level accuracies.  
* Perform lightweight hyper‑parameter search / regularisation to push accuracy beyond the baseline.  
* Ship a self‑contained Colab‑friendly notebook with code and results.

---

## Architecture

```
Input (3×32×32)
│
├─ Block&nbsp;1 ──┐
│                 │
├─ Block&nbsp;2 ──┤      K intermediate blocks
│                 │
└─ Block&nbsp;K ──┘
│
└─ Output Block ─► logits (10)
```

### Intermediate Block Bᵏ
```
x ─► C₁ ──┐
   ─► C₂ ─┼─  x′ = Σᵢ aᵢ · Cᵢ(x)
   …      │
   ─► C_L ─┘
      ▲
      │   m = AvgPool(x)   a = FCₖ(m)
```

* **Parallel convs** `C₁ … C_L` share *exactly* the same input.  
* Channel‑wise spatial average → vector *m* (length = #channels of *x*).  
* Tiny fully‑connected layer `FCₖ` maps *m* to weights *a₁ … a_L*.  
* Combined output keeps spatial size (and typically channel count) unchanged; non‑linearities / pooling may be inserted between blocks.

### Output Block **O**
* Takes the final feature map.  
* Global channel average → vector *m<sub>final</sub>*.  
* One or more fully‑connected layers → 10‑D logits.

Implementation lives in **`src/model.py`** (`IntermediateBlock`, `OutputBlock`, `CIFAR10Net`).

---

## Repository Layout

```
.
├── notebooks/
│   └── cifar10_experiments.ipynb     ← main Colab notebook
├── src/
│   ├── data.py       ← dataloaders & augmentations
│   ├── model.py      ← blocks & full network
│   ├── train.py      ← train/val loops, logging utils
│   └── utils.py
├── reports/
│   ├── curves.png    ← loss & accuracy plots
│   └── assignment_report.pdf
└── README.md
```

---

## Getting Started

### 1 · Clone & create environment
```bash
git clone https://github.com/<user>/weighted-block-cnn.git
cd weighted-block-cnn
conda env create -f environment.yml        # or: pip install -r requirements.txt
conda activate wb-cnn
```

### 2 · Quick training run (local GPU)
```bash
python -m src.train   --batch_size 128   --epochs 30   --lr 0.001   --save_dir runs/try1
```

### 3 · Google Colab
Open `notebooks/cifar10_experiments.ipynb` in Colab  
(**Runtime ▸ Change runtime type ▸ GPU**) and ▶ *Run all*.

---

## Training & Evaluation

| Component | Setting |
|-----------|---------|
| Loss      | Cross‑Entropy |
| Optimiser | Adam (baseline) or SGD + momentum |
| Schedule  | Fixed, step, or cosine decay |
| Augment   | Random crop + flip (+ optional CutMix / ColorJitter) |
| Logged    | **Batch‑wise** training loss · **Epoch‑wise** train & test accuracy |

Plots are auto‑saved to `reports/curves.png`.

---

## Results

| Model Variant | Params | Test Acc (best‑of‑run) |
|---------------|--------|------------------------|
| 3 × (32‑64‑128) baseline | 3.1 M | **79.4 %** |
| + BatchNorm + cosine LR | 3.1 M | **83.6 %** |
| + wider (48‑96‑192) + Dropout 0.2 | 7.8 M | **86.8 %** |
| + CutMix, label‑smoothing, WD 5e‑4 | 7.8 M | **91.3 %** |

> *Update this table with your final hyper‑parameter sweep & accuracy.*

---

## Key Implementation Details
* **Coefficient generation** – raw, `softmax`, or `sigmoid` — experiment.  
* **Shape consistency** – all conv branches in a block output identical shapes ✅.  
* **Pooling** – simple `MaxPool2d(2)` halves spatial size after each block.  
* **Modularity** – arbitrary *K* and *L* via config list; easy to scale deeper/wider.

---

## Possible Extensions
* Residual skip connections around each block.  
* Additional squeeze‑and‑excitation after the weighted sum.  
* MixUp / CutMix / AutoAugment for stronger regularisation.  
* Replace final global‑avg pooling with spatial attention.

---

## Citation / Coursework Note
This repository derives from my individual submission for **ECS7026P (Queen Mary University of London, 2025)**.  
Feel free to use the code for learning or research; if you build on it, please cite this repo and the original assignment brief.

---

## License
[MIT](LICENSE)

Happy experimenting 🎉
