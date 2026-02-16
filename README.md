# Linear Combination of Unitaries (LCU)

This repo is a working, readable implementation of the **Linear Combination of Unitaries (LCU)** construction in Qiskit using the standard **PREPARE → SELECT → UNPREPARE** pattern, with **postselection** on the ancilla register.

It is written as a learning friendly reference: you can plug in coefficients and Pauli strings (or circuits / matrices), build the LCU unitary, run it on Aer, and compare **empirical** vs **theoretical** postselection success probability. :contentReference[oaicite:0]{index=0}

---

## What this repo implements

You start with an operator written as a linear combination of unitaries:
$$
A = \sum_{j=0}^{L-1} \alpha_j U_j
$$
where $\alpha_j \in \mathbb{C}$ and each $U_j$ is unitary.

Since a quantum circuit must be unitary, you do not implement $A$ directly. Instead, you build a larger unitary $W$ such that when you apply it to $|0\rangle_{\text{anc}}|\psi\rangle_{\text{tgt}}$ and postselect the ancilla on $|0\cdots 0\rangle$, the target register behaves like $A|\psi\rangle$ up to a known scaling:
$$
\left(\langle 0\cdots 0|_{\text{anc}} \otimes I\right)\, W \, \left(|0\cdots 0\rangle_{\text{anc}} \otimes |\psi\rangle_{\text{tgt}}\right) \propto A|\psi\rangle
$$

---

## Algorithm, as implemented

### 1. Coefficient preprocessing

Each coefficient is split into magnitude and phase:
$$
\alpha_j = |\alpha_j| e^{i\phi_j}
$$

Define:
$$
\lambda = \sum_j |\alpha_j|,\quad p_j = \frac{|\alpha_j|}{\lambda}
$$

The ancilla amplitude vector is built using:
$$
\beta_j = \sqrt{p_j} = \sqrt{\frac{|\alpha_j|}{\lambda}}
$$
and padded to size $2^m$ where $m=\lceil \log_2 L \rceil$. :contentReference[oaicite:1]{index=1}

---

### 2. PREPARE oracle

PREPARE maps the all zero ancilla state to an index superposition encoding the coefficient magnitudes:
$$
\text{PREPARE}\,|0\cdots 0\rangle = \sum_{j} \sqrt{p_j}\,|j\rangle
$$

This repo contains two variants:

- **Baseline**: uses Qiskit `StatePreparation` directly for PREPARE and PREPARE$^\dagger$. :contentReference[oaicite:2]{index=2}  
- **Decomposed (educational)**: inlines PREPARE and PREPARE$^\dagger$ so you can *see* their decomposition, and targets a restricted gate set (notably a special mode aiming for only `ry` and `cx`). :contentReference[oaicite:3]{index=3}

Important note: preparing an arbitrary complex state using only $\{ry, cx\}$ is not possible in general. The decomposed variant explicitly enforces **real amplitude** state preparation and will raise an error if nontrivial phases are present. :contentReference[oaicite:4]{index=4}

---

### 3. SELECT oracle

SELECT applies the right unitary conditioned on the ancilla state:
$$
\text{SELECT} = \sum_j |j\rangle\langle j| \otimes e^{i\phi_j} U_j
$$

Implementation strategy used here:

1. Loop over each term $j$
2. Flip ancilla bits so that the branch $|j\rangle$ maps to $|11\cdots 1\rangle$
3. Apply a controlled phase $e^{i\phi_j}$ on that branch
4. Apply the controlled $U_j$
5. Undo the flips :contentReference[oaicite:5]{index=5}

Supported $U_j$ formats:
- Pauli strings like `"XYZZY"` (fast path)
- `QuantumCircuit`
- numpy matrices
- callables for custom behavior :contentReference[oaicite:6]{index=6}

---

### 4. UNPREPARE and postselection

The full LCU unitary is:
$$
W = (\text{PREPARE}^\dagger \otimes I)\, \text{SELECT}\, (\text{PREPARE} \otimes I)
$$

Finally, measure the ancilla and postselect on:
$$
|0\cdots 0\rangle_{\text{anc}}
$$

---

## Success probability (empirical and theoretical)

For the postselected construction, the success probability is:
$$
p_{\text{success}} = \frac{\|A|\psi\rangle\|^2}{\lambda^2}
$$

This repo reports:

- **Empirical** $p_{\text{success}}$: fraction of shots where ancilla measured `0...0`
- **Theoretical** $p_{\text{success}}$: computed by explicitly forming $A|\psi\rangle$ when the unitaries are representable as matrices (Pauli strings or explicit gates) :contentReference[oaicite:7]{index=7}

When $\lambda$ is large, postselection becomes unlikely. In full scale algorithms this is typically handled with amplitude amplification, which is not implemented here.

---

## Repository structure

- `main.py`  
  Baseline LCU implementation using `StatePreparation` blocks for PREPARE / PREPARE$^\dagger$. Includes parsing helpers, controlled Pauli string implementation, circuit builder, Aer sampling, and empirical vs theoretical probability reporting. :contentReference[oaicite:8]{index=8}

- `main_stateprep_decomposed.py`  
  Variant that **inlines** state preparation and attempts to **decompose** PREPARE and PREPARE$^\dagger$ down to a very restricted gate set (aiming for `ry` and `cx`). This is primarily for circuit inspection and learning. It enforces real statevectors for that restriction. :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}

---

## Installation

```bash
pip install qiskit qiskit-aer matplotlib numpy
