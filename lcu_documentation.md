# Linear Combination of Unitaries (LCU)

This repository implements the Linear Combination of Unitaries (LCU) technique in Qiskit, using the standard **PREPARE–SELECT–UNPREPARE** construction and **postselection** on the ancilla register to realize a generally non unitary operator on a quantum state.

The code is written to be both a working reference implementation and a learning tool. It includes utilities to parse input states, build controlled Pauli string unitaries, construct the full LCU circuit, and empirically estimate postselection success probability. A theoretical success probability is also computed when the unitaries are provided as Pauli strings or explicit matrices.

## What this repo does

Given an operator written as a linear combination of unitaries

$$
A = \sum_{j=0}^{L-1} \alpha_j U_j
$$

where \( \alpha_j \in \mathbb{C} \) and \(U_j\) are unitary operators, the implementation constructs a larger unitary \(W\) such that, when applied to \(|0\rangle_{\text{anc}}|\psi\rangle_{\text{tgt}}\) and postselecting the ancilla on \(|0\rangle\),

$$
\langle 0| W |0\rangle |\psi\rangle \propto A|\psi\rangle
$$

This is the core LCU idea. You do not directly apply \(A\). You embed it inside a unitary and recover the action of \(A\) probabilistically via postselection.

---

## Algorithm overview

### 1. Coefficient preprocessing

Each coefficient is split into magnitude and phase:

$$
\alpha_j = |\alpha_j| e^{i\phi_j}
$$

Define the normalization factor

$$
\lambda = \sum_{j} |\alpha_j|
$$

and a probability distribution

$$
p_j = \frac{|\alpha_j|}{\lambda}
$$

### 2. PREPARE oracle

Prepare an ancilla register state encoding the coefficient magnitudes:

$$
\text{PREPARE}\,|0\rangle = \sum_j \sqrt{p_j}\,|j\rangle
$$

In the implementation, this is done using Qiskit `StatePreparation` over the ancilla register. The amplitudes are explicitly constructed as:

$$
\beta_j = \sqrt{\frac{|\alpha_j|}{\lambda}}
$$

and padded to length \(2^m\) where \(m=\lceil \log_2 L\rceil\).

### 3. SELECT oracle

Conditioned on the ancilla state \(|j\rangle\), apply the corresponding unitary with its phase:

$$
\text{SELECT} = \sum_j |j\rangle\langle j| \otimes e^{i\phi_j} U_j
$$

In code, this is implemented by iterating through each term \(j\), mapping the ancilla basis \(|j\rangle\) to \(|11\ldots 1\rangle\) with X flips, then applying a controlled phase and a controlled unitary, then undoing the flips.

Supported unitaries:
- Pauli strings like `"XYZZY"` (fast path)
- `QuantumCircuit` objects
- numpy matrices
- callables for custom controlled behavior

### 4. UNPREPARE and postselection

Apply PREPARE† to uncompute the ancilla:

$$
W = (\text{PREPARE}^\dagger \otimes I)\, \text{SELECT}\, (\text{PREPARE} \otimes I)
$$

Then measure the ancilla register and **postselect** on outcome:

$$
|0\ldots 0\rangle_{\text{anc}}
$$

Only these shots are considered “successful” realizations of \(A|\psi\rangle\) up to scaling.

---

## Success probability

For the postselected construction, the success probability is:

$$
p_{\text{success}} = \frac{\|A|\psi\rangle\|^2}{\lambda^2}
$$

This repo provides:
- **Empirical estimate** from sampling counts on Aer
- **Theoretical value** computed by explicitly forming \(A|\psi\rangle\) when all \(U_j\) are known as matrices or Pauli strings

When \(\lambda\) is large, postselection becomes unlikely. In practical algorithms, this is typically handled with **amplitude amplification**, which is not implemented here, but the code is structured to make that extension straightforward.

---

## Repository structure

- `lcu.py`  
  Main implementation. Includes:
  - state parsing helpers
  - unitary parsing helpers
  - controlled Pauli string application
  - LCU class for building and running circuits

- `examples/`  
  Example configurations and experiments. Each example instantiates the `LCU` class with coefficients, unitaries, and an initial state string.

---

## Installation

This code uses Qiskit and Qiskit Aer.

```bash
pip install qiskit qiskit-aer matplotlib numpy
