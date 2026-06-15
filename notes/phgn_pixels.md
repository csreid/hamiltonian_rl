---
title: Port-Hamiltonian World Models from Pixels
author: Cameron Reid
---

# Motivation

## The Problem

* Physical systems have rich structure: energy, dissipation, control ports
* Black-box models (e.g. DreamerV3) ignore this — data-hungry and opaque
* **Goal:** learn a world model that is physically interpretable *and* sample-efficient

## Port-Hamiltonian Systems

A port-Hamiltonian system evolves as:

$$\dot{z} = (J - R)\nabla\mathcal{H}(z) + Bu$$

* $J$ — symplectic (energy-routing) matrix: $J = -J^\top$
* $R$ — dissipation matrix: $R = R^\top \succeq 0$
* $\mathcal{H}$ — Hamiltonian (total energy)
* $B$ — control input matrix

Encodes energy conservation, dissipation, and controllability in the *structure* of the ODE

## Why From Pixels?

* Prior work (port-Hamiltonian Neural ODEs) assumes direct access to $(q, p)$
* In practice, sensors give images, not joint angles and momenta
* **Challenge:** discover a phase space *and* port-Hamiltonian structure from image sequences

# Approach

## Overview

Two-phase training pipeline:

1. **Phase 1** — learn a compact latent representation $h_t$ from image sequences (recurrent autoencoder)
2. **Phase 2** — learn a normalizing flow $\Phi: h \to (q, p)$ such that Hamiltonian dynamics hold in $(q, p)$ space

Inference: encode a few context frames $\to$ map to phase space $\to$ roll out dynamics $\to$ invert back to pixels

## Phase 1: Recurrent Autoencoder

* Causal LSTM over frame sequence $x_0, \dots, x_T$ produces hidden state $h_t$ at each step
* $h_t$ is projected to a latent $(q, p)$ pair via a learned MLP $f_\psi$
* Trained on two pixel-space objectives:
  * **Reconstruction:** decode $f_\psi(h_t)$ back to $x_t$
  * **Next-frame prediction:** decode $f_\psi(h_t)$ to $x_{t+1}$ given action $u_t$
* Also includes a KL regularizer (VAE-style) and a temporal metric regularizer

## Phase 1: Temporal Metric Regularizer

* Problem: $h_t$ representations could be arbitrarily permuted in latent space
* Fix: penalize pairs $(h_{t_1}, h_{t_2})$ that are *closer* in $h$-space than expected from $|t_1 - t_2|$
* Forces $h_t$ to be temporally ordered in latent space — smoother manifold for Phase 2

## Phase 2: Why a Normalizing Flow?

$h_t$ already encodes the phase space implicitly:

* Reconstructing $x_t$ requires position information ($q$)
* Predicting $x_{t+1}$ requires velocity information ($p$)

But this implicit phase space has no reason to respect port-Hamiltonian structure — it's just whatever the LSTM found convenient.

**Fix:** learn a normalizing flow $\Phi: h \to (q, p)$

* A lossless change of coordinates from the LSTM's arbitrary phase space to one that is *induced* to follow port-Hamiltonian dynamics
* Invertible by construction: $\Phi^{-1}(q, p) = h$ lets us decode back to pixels through the Phase 1 network

## Phase 2: Training Objectives

Both objectives share one $\Phi$ forward pass over the full sequence:

**Teacher-forced** — one RK4 step from every $h_t$, batched:

$$\mathcal{L}_\text{tf} = \text{MSE}\!\left(\Phi^{-1}\!\left(\text{RK4}(\Phi(h_t), u_t)\right),\; h_{t+1}\right)$$

**Closed-loop** — autonomous rollout from seed $h_k$ for $\ell$ steps:

$$\mathcal{L}_\text{cl} = \frac{1}{\ell}\sum_{i=1}^{\ell}\text{MSE}\!\left(\Phi^{-1}\!\left(\text{RK4}^i(\Phi(h_k), u)\right),\; h_{k+i}\right)$$

* $\ell$ starts small and advances via a **performance-gated curriculum**: only grow once $\mathcal{L}_\text{cl}$ is sufficiently low

## Learned Structure: Split Learning Rates

$J$, $R$, $B$, $\mathcal{H}$, and $\Phi$ are all learned jointly, but with two separate learning rates:

* $\mathcal{H}$ and $\Phi$ use a small LR (e.g. $10^{-4}$)
* $J$, $R$, $B$ use a much larger LR (e.g. $10^{-2}$)

**Why:** structural matrices are initialized near zero, so their gradient signal is negligible early in training — $\nabla\mathcal{H}$ is small, so $(J-R)\nabla\mathcal{H} \approx 0$ regardless of $J$ and $R$. Without a higher LR they never escape this regime and the dynamics reduce to $\dot{z} \approx Bu$.

## Learned Structure: Frobenius Regularization

$$\mathcal{L}_\text{struct} = \lambda_s \left(\|J\|_F^2 + \|R\|_F^2\right)$$

**Why:** without regularization, $J$ and $R$ grow to large values over long training runs. Large structural matrices amplify $\nabla\mathcal{H}$ in the dynamics step, producing large and chaotic phase-space trajectories on held-out data. The regularizer penalizes the optimizer for finding solutions that rely on extreme matrix values.

## Learned Structure: Logdet Regularizer

$$\mathcal{L}_\text{logdet} = \lambda_\Phi \cdot \mathbb{E}\!\left[(\log|\det J_\Phi|)^2\right]$$

**Why:** a normalizing flow can in principle collapse the entire latent space to a small region (large negative log-det) or expand it without bound (large positive log-det). Either pathology makes the inverse $\Phi^{-1}$ poorly conditioned. Penalizing the squared log-determinant keeps $\Phi$ near-volume-preserving, which stabilizes both the forward and inverse passes.

## Inference

Given a short context of frames:

1. Encode context with Phase 1 LSTM: $h_0, \dots, h_{k}$
2. Map to phase space: $(q_k, p_k) = \Phi(h_k)$
3. Roll out port-Hamiltonian dynamics for $T$ steps: $(q_k, p_k) \to \cdots \to (q_{k+T}, p_{k+T})$
4. Invert back: $h_{k+i} = \Phi^{-1}(q_{k+i}, p_{k+i})$
5. Decode to pixels via Phase 1 $f_\psi$ and decoder

# Current Status

## What's Working

* Phase 1 autoencoder learns good reconstructions and next-frame predictions
* Phase 2 teacher-forced loss converges reliably

## Current Issue: Overfitting

* Training losses (both $\mathcal{L}_\text{tf}$ and $\mathcal{L}_\text{cl}$) converge well
* Held-out dreamed rollouts look chaotic — dynamics don't generalize
* Structural matrices ($J$, $R$) grow large and produce high-variance gradients in the dynamics step
* Added Frobenius regularization on $J, R$ as a first fix; added train/val loss logging to confirm the gap
* Structural matrices don't look right: with dissipation set to zero in data collection, $R$ should converge to zero, but instead persistently develops one large eigenvalue concentrated at a single latent dimension; $J$ develops correspondingly large entries at the same coordinates, possibly attempting to cancel the spurious dissipation

## Next Steps

* Confirm train vs. val gap quantitatively in TensorBoard
* Investigate root cause:
  * Phase 2 overfitting to the cached $h_t$ representations?
