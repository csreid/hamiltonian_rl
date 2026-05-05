---
title: Updates & Notes
author: Cameron
---

# Structured/Physically-Grounded Generative Networks

## Overview

- Hamiltonian Generative networks [@toth2019hamiltonian]
- Port-Hamiltonian Neural ODEs for Robot Dynamics

## HGN

- Input: images $\in \mathbb{R}^{3 \times h \times w}$
- Map to phase space $(p,q) \in \mathbb{R}^n$
- Learn the Hamiltonian $\mathcal{H}(p,q)$
  - Perhaps specifically as $\mathcal{H}(p,q) = T(p,q) + V(q)$
- Simulate forward by integrating $\mathcal{H}$ forward

## HGN

- Supervision is pixel decoding
- Energy-conserving, physically-grounded.
- [SHO results](http://localhost:6006/?darkMode=true#images&runSelectionState=eyJNYXIwNV8yMS00NS00OF9VSUNST2JvdGljc19zaG9fYmVuY2htYXJrIjpmYWxzZSwiTWFyMDVfMjItMDYtMjZfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA1XzIyLTA4LTM5X1VJQ1JPYm90aWNzX3Nob19pbWFnZXMiOmZhbHNlLCJNYXIwNV8yMy0xMS0yM19VSUNST2JvdGljc19zaG9faW1hZ2VzIjpmYWxzZSwiTWFyMDVfMjMtMTMtMDJfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA1XzIzLTE3LTQzX1VJQ1JPYm90aWNzX3Nob19pbWFnZXMiOmZhbHNlLCJNYXIwNV8yMy0zMi0zNV9VSUNST2JvdGljc19zaG9faW1hZ2VzIjpmYWxzZSwiTWFyMDVfMjMtMzUtMDFfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA1XzIzLTM2LTU0X1VJQ1JPYm90aWNzX3Nob19pbWFnZXMiOmZhbHNlLCJNYXIwNV8yMy0zNy0yN19VSUNST2JvdGljc19zaG9faW1hZ2VzIjpmYWxzZSwiTWFyMDVfMjMtMzgtMDVfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA1XzIzLTM5LTAyX1VJQ1JPYm90aWNzX3Nob19pbWFnZXMiOmZhbHNlLCJNYXIwNV8yMy00MC0xNV9VSUNST2JvdGljc19zaG9faW1hZ2VzIjpmYWxzZSwiTWFyMDVfMjMtNDctMzZfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA1XzIzLTQ5LTE5X1VJQ1JPYm90aWNzX3Nob19pbWFnZXMiOmZhbHNlLCJNYXIwNV8yMy01NC01MV9VSUNST2JvdGljc19zaG9faW1hZ2VzIjpmYWxzZSwiTWFyMDVfMjMtNTUtNTZfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA2XzAwLTA0LTQxX1VJQ1JPYm90aWNzX3Nob19pbWFnZXMiOmZhbHNlLCJNYXIwNl8wMC0wNy0yOF9VSUNST2JvdGljc19zaG9faW1hZ2VzIjpmYWxzZSwiTWFyMDZfMTMtMTMtMTVfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA2XzEzLTE2LTExX1VJQ1JPYm90aWNzX3Nob19pbWFnZXMiOmZhbHNlLCJNYXIwNl8xMy0yMC0zNF9VSUNST2JvdGljc19zaG9faW1hZ2VzIjpmYWxzZSwiTWFyMDZfMTMtMjYtMzFfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA2XzEzLTI3LTM1X1VJQ1JPYm90aWNzX3Nob19pbWFnZXMiOmZhbHNlLCJNYXIwNl8xMy00Mi0wN19VSUNST2JvdGljc19zaG9faW1hZ2VzIjpmYWxzZSwiTWFyMDZfMTMtNTEtMjhfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA2XzE2LTAyLTUxX1VJQ1JPYm90aWNzX3Nob19pbWFnZXMiOmZhbHNlLCJNYXIwNl8xNi0wMy0yMV9VSUNST2JvdGljc19zaG9faW1hZ2VzIjpmYWxzZSwiTWFyMDZfMTYtMDktMDhfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA2XzE2LTE2LTQ2X1VJQ1JPYm90aWNzX3Nob19pbWFnZXMiOmZhbHNlLCJNYXIwNl8xNi0yMi0xN19VSUNST2JvdGljc19zaG9faW1hZ2VzIjpmYWxzZSwiTWFyMDZfMTgtMzUtMTZfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA2XzE4LTQwLTQzX1VJQ1JPYm90aWNzX3Nob19pbWFnZXMiOmZhbHNlLCJNYXIwNl8xOC01Mi0zN19VSUNST2JvdGljc19zaG9faW1hZ2VzIjpmYWxzZSwiTWFyMDZfMTgtNTMtMDZfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA2XzE5LTExLTAxX1VJQ1JPYm90aWNzX3Nob19pbWFnZXMiOmZhbHNlLCJNYXIwNl8xOS0xNy0wNl9VSUNST2JvdGljc19zaG9faW1hZ2VzIjpmYWxzZSwiTWFyMDZfMTktMTktMDVfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA2XzE5LTIxLTI5X1VJQ1JPYm90aWNzX3Nob19pbWFnZXMiOmZhbHNlLCJNYXIwNl8xOS0yNS0yNl9VSUNST2JvdGljc19zaG9faW1hZ2VzIjpmYWxzZSwiTWFyMDZfMTktMjYtNDZfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA2XzE5LTI4LTI3X1VJQ1JPYm90aWNzX3Nob19pbWFnZXMiOmZhbHNlLCJNYXIwNl8xOS0zOC0wOF9VSUNST2JvdGljc19zaG9faW1hZ2VzIjpmYWxzZSwiTWFyMDZfMTktMzktNThfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA2XzE5LTQ4LTAxX1VJQ1JPYm90aWNzX3Nob19pbWFnZXMiOmZhbHNlLCJNYXIwNl8xOS01My00OV9VSUNST2JvdGljc19zaG9faW1hZ2VzIjpmYWxzZSwiTWFyMDZfMjAtMTItMTRfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlc19yZWN1cnJlbnQiOmZhbHNlLCJNYXIwNl8yMC0xNS0wM19VSUNST2JvdGljc19zaG9faW1hZ2VzX3N0YWNrZWQiOmZhbHNlLCJNYXIwNl8yMS0yMS0yNF9VSUNST2JvdGljc19oZ25fb3JnX3Nob19pbWFnZXMiOmZhbHNlLCJNYXIwNl8yMS0yMi0yMl9VSUNST2JvdGljc19oZ25fb3JnX3Nob19pbWFnZXMiOmZhbHNlLCJNYXIwNl8yMS0yNS0xNl9VSUNST2JvdGljc19oZ25fb3JnX3Nob19pbWFnZXMiOmZhbHNlLCJNYXIwNl8yMS0zMi01NF9VSUNST2JvdGljc19oZ25fb3JnX3Nob19pbWFnZXMiOmZhbHNlLCJNYXIwN18xOC0yNC0xN19VSUNST2JvdGljc19zaG9faW1hZ2VzX3JlY3VycmVudCI6ZmFsc2UsIk1hcjA3XzE4LTI1LTQ0X1VJQ1JPYm90aWNzX3Nob19pbWFnZXNfcmVjdXJyZW50IjpmYWxzZSwiTWFyMDdfMTgtMjgtMzBfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlc19yZWN1cnJlbnQiOmZhbHNlLCJNYXIwN18xOC0zMi00OV9VSUNST2JvdGljc19zaG9faW1hZ2VzX3JlY3VycmVudCI6ZmFsc2UsIk1hcjA3XzE4LTMzLTQyX1VJQ1JPYm90aWNzX3Nob19pbWFnZXNfcmVjdXJyZW50IjpmYWxzZSwiTWFyMDdfMTktMDYtMzlfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlc19yZWN1cnJlbnQiOmZhbHNlLCJNYXIwN18xOS0xMS00MV9VSUNST2JvdGljc19zaG9faW1hZ2VzX3JlY3VycmVudCI6ZmFsc2UsIk1hcjA3XzE5LTIxLTQzX1VJQ1JPYm90aWNzX3Nob19pbWFnZXNfcmVjdXJyZW50IjpmYWxzZSwiTWFyMDdfMTktMjUtMTlfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlc19yZWN1cnJlbnQiOmZhbHNlLCJNYXIwN18xOS0zNS0wOV9VSUNST2JvdGljc19zaG9faW1hZ2VzX3JlY3VycmVudCI6ZmFsc2UsIk1hcjA3XzE5LTQ4LTE5X1VJQ1JPYm90aWNzX3Nob19pbWFnZXNfcmVjdXJyZW50IjpmYWxzZSwiTWFyMDdfMjAtMTktNTlfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlc19yZWN1cnJlbnQiOmZhbHNlLCJNYXIwN18yMC00Ny0zNF9VSUNST2JvdGljc19zaG9faW1hZ2VzX3JlY3VycmVudCI6ZmFsc2UsIk1hcjA3XzIwLTQ5LTE5X1VJQ1JPYm90aWNzX3Nob19pbWFnZXNfcmVjdXJyZW50IjpmYWxzZSwiTWFyMDdfMjAtNTItNThfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlc19yZWN1cnJlbnQiOmZhbHNlLCJNYXIwN18yMS0wMS0wN19VSUNST2JvdGljc19zaG9faW1hZ2VzX3JlY3VycmVudCI6ZmFsc2UsIk1hcjA3XzIxLTQxLTE2X1VJQ1JPYm90aWNzX3Nob19pbWFnZXNfcmVjdXJyZW50IjpmYWxzZSwiTWFyMDdfMjEtNTEtMTNfVUlDUk9ib3RpY3Nfc2hvX2ltYWdlc19yZWN1cnJlbnQiOmZhbHNlLCJNYXIwOF8wMS0xMy0xN19VSUNST2JvdGljc19zaG9faW1hZ2VzX3JlY3VycmVudCI6ZmFsc2UsIk1hcjA4XzE5LTI0LTM3X1VJQ1JPYm90aWNzX2hnbl9vcmdfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA4XzE5LTI1LTA2X1VJQ1JPYm90aWNzX2hnbl9vcmdfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA4XzE5LTQ4LTUyX1VJQ1JPYm90aWNzX2hnbl9vcmdfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA4XzE5LTQ5LTU4X1VJQ1JPYm90aWNzX2hnbl9vcmdfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA4XzE5LTUxLTQzX1VJQ1JPYm90aWNzX2hnbl9vcmdfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA4XzE5LTU0LTE0X1VJQ1JPYm90aWNzX2hnbl9vcmdfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA4XzE5LTU4LTUxX1VJQ1JPYm90aWNzX2hnbl9vcmdfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA4XzIwLTIwLTQ3X1VJQ1JPYm90aWNzX2hnbl9vcmdfc2hvX2ltYWdlcyI6ZmFsc2UsIk1hcjA4XzIyLTEyLTI5X1VJQ1JPYm90aWNzX2hnbl9vcmdfc2hvX2ltYWdlcyI6dHJ1ZSwiTWFyMDhfMjItMzQtMTdfVUlDUk9ib3RpY3NfaGduX29yZ19zaG9faW1hZ2VzIjpmYWxzZX0%3D)
  - (Some generalization issues)

# Port-Hamiltonian Neural ODE for Robots

## Dynamics

- Input is ~$(p,q)$ directly (computed from ground-truth position/velocity/etc data)
- Learn a momentum:

$$
\dot{p} = \text{ad}*(p) - D \cdot M^{-1} p - \frac{\partial V}{\partial q} + B \cdot u
$$

- And position:

$$\dot{q} = M^{-1}p$$

- $M^{-1}$, $D$, $V$, $B$ are all learned as neural networks

## Notes

- Extends to dissipation and control input
- Assumes highly informative/low-dimensional state input to compute initial $q, p$ directly.

# Plan

## Somewhere in between

- Relax the assumption of perfect information
- Map to phase space as in [@toth2019hamiltonian], but use the port-Hamiltonian formulation of [@duong2024port]

## Progression

1. HGN
2. HGN w/ dissipation (damped SHO?)
3. Full PH with cartpole
4. PH with Racecar (or similar)
5. PH with our car

# 3/16

## Damped SHO

![](images/damped.gif)

## Damped SHO

![](images/corr.png)

## Issues

* HGN looks at full trajectory to get p0,q0; we need live/online phase space
* Maybe LSTM is enough?
* Bi-directional for the past?
	* But then we have a limited history window
* Ideally, we maintain a good estimate of position in phase space from images; relevant history is summarized in $h_t$ from the RNN

## In Progress:
* Recurrent version of HGN
	* Just fixed a memory leak, should be good soon. Looks promising
	* Still looks at the whole trajectory, from end to beginning, to get p0,q0
* PHGN for cartpole, compare to PPO
	* Currently just looking at previous 4 frames, very bad performance
* Better recurrent architecture?

# 3/30

## Updates

* Recurrent HGN + Dissipative HGN are good
* PPO works
* Full port-Hamiltonian works(?) for CartPole

## Port Hamiltonian Design

1. Process some sequence of transitions with LSTM
1. Choose $h_k$ (where $k$ is some warmup context length)
1. Map $h_k$ to $(q, p)_k$
1. Integrate Hamiltonian forward to get $(q, p)_{k \dots t}$
1. Supervise decoding LSTM state to this sequence

## Current Issues

1. Pixel reconstruction is bad and not contributing.
	* Always just a white square.
1. Mysterious collapse in performance

## Collapse

![](images/collapse.png)

# 4/13

## Change of Plans

* Full CartPole + MPPI pipeline was too ambitious for now
* Backed off to the simpler problem: **learn phase space + reconstruction from pixels**
  * Same framing as the SHO work, but for Pendulum
  * Wrapped `Pendulum-v1` to add damping

## Architecture

1. Bidirectional LSTM over the **full trajectory** $\rightarrow (q_0, p_0)$
2. Integrate Hamiltonian dynamics forward with **RK4**
3. Decode phase-space trajectory back to pixels

## Results

![](images/pendulum_recon.gif)

GT (left) vs. reconstruction (right)

## Why Pendulum?

* Trying to eliminate as many variables as possible
* Pendulum has a **separable Hamiltonian**: $\mathcal{H}(q,p) = T(p) + V(q)$
  * Stronger inductive bias than CartPole
  * $T$ and $V$ can be learned independently
* CartPole is non-separable — removing that complexity first

## Dissipation Structure

![](images/R_eigenvalues.png)

One large eigenvalue, 15 near zero — model correctly concentrates dissipation in one mode

## Concerns

* Results might be **too good** --- want to check longer Hamiltonian rollouts to ensure it generalizes

# 4/27

## Shift: Pixels $\rightarrow$ Phase Space

* Goal: confirm the port-Hamiltonian dynamics model can actually learn before adding pixel encoding back
* I found lots of stuff

## Problem: Structural Matrices Not Learning

* Analytic rollouts were ~static
* Suspected: $b$ initialized near zero with a small lr $\rightarrow$ contributes nothing to the Hamiltonian $\rightarrow$ no gradient signal
* Tested by fixing them to the ground truth, $b = 3$,  $J = [[0, 1],[-1, 0]]$, $R = [[0, 0], [0, d]]$ (where $d$ is configurable damping factor) and training just $\mathcal{H}$
	* model learned well, confirmed diagnosis
* Fix: **split learning rates** --- small LR for $\mathcal{H}$, much larger LR for structural matrices ($J$, $R$, $B$)
* Re-enabled joint learning of all structural matrices

## Problem: Model Can't Learn Long Sequences

* With structural matrices learning, model handled short rollouts fine but collapsed beyond $\sim 25$ steps
* Suspected: gradients dominated by noise from a bad long rollout early in training
* Added **curriculum learning**: start with short sequences, gradually increase length
* Linear schedule advanced too fast
* Switched to **performance-gated curriculum**: only advance sequence length once loss is sufficiently low

## Problem: Model (Still) Can't Learn Long Sequences

* Curriculum helped, but didn't fix the issue
* Suspected: we detach the gradients in the dynamics rollout, so gradients re: long-term dependencies are incorrect
* Fix: don't detach those, let the Hessian come through

## Current status:

* Everything is learning really well
* Structural matrices have good/coherent values
* Predicted rollouts look really good
* Back to pixel space?

# 5/4

## Back to Pixel Space (Apr 29)

* Backported all the fixes discovered in the state-based model back to the pixel-space model
* Pixel-space model is now just responsible for mapping to a good phase space; dynamics happen in phase space as before

## Problem: Hamiltonian Drift from LSTM (May 1)

* The LSTM encoder produces a $q$ from the image sequence, but Hamiltonian rollouts were drifting away from it
* Fix: added a **dynamic/LSTM alignment penalty** --- an extra loss term penalizing the encoder when the Hamiltonian dynamics rollout diverges from the $q$ the LSTM learned
* Idea: force the Hamiltonian to stay "on-manifold" with respect to what the encoder actually sees

## Problem: Consistency Loss Blows Up (May 3)

* The alignment/consistency loss was blowing up exponentially when propagated across the full rollout
* Fix: restrict the consistency loss to **just one integrator step** into the future rather than rolling out gradients through the entire sequence
* This keeps the loss bounded without sacrificing the core signal

## Core Problem: $f_\psi$ Loses Information (May 3)

* Diagnosis: the LSTM hidden state $h_t$ captures the environment really well --- but the mapping $f_\psi: h \to (q, p)$ is a plain MLP and is lossy
* When the Hamiltonian is then rolled out, it produces phase-space coordinates the decoder can't reconstruct from, because $f_\psi$ threw away information

## Proposed Fix: Normalizing Flow for $h \to (q, p)$

* Replace $f_\psi$ with a **normalizing flow**
* A normalizing flow is invertible by construction, so all information in $h$ survives the mapping to $(q, p)$
* Crucially: the flow goes both ways --- we can map $h \to (q, p)$ for Hamiltonian rollouts *and* map $(q, p) \to h$ to feed back into the decoder
* This should close the loop: Hamiltonian dynamics operate in a coherent phase space, and we can always invert back to the LSTM's representation for decoding
