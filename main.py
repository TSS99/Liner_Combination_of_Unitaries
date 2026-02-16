
# All the imports

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from typing import List, Union, Callable, Optional

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _is_binary_string(s: str) -> bool:
    return len(s) > 0 and all(c in "01" for c in s)

def _normalize_state(vec: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    vec = np.asarray(vec, dtype=complex)
    nrm = np.linalg.norm(vec)
    if nrm < atol:
        raise ValueError("Statevector has (near) zero norm.")
    return vec / nrm

def parse_initial_state(initial_state: Union[str, List, np.ndarray]) -> np.ndarray:
    """
    Supported:
      - list/np.array statevector
      - '|0>', '|1>', '|+>', '|->', '|+i>', '|-i>'
      - '0101' (basis)
      - '0001+0010+1111' (equal superposition)
    """
    if isinstance(initial_state, (list, np.ndarray)):
        return _normalize_state(np.array(initial_state, dtype=complex))

    s = str(initial_state).strip().replace(" ", "")
    if s.startswith("|") and s.endswith(">"):
        s = s[1:-1]

    named = {
        "0":  np.array([1, 0], dtype=complex),
        "1":  np.array([0, 1], dtype=complex),
        "+":  np.array([1, 1], dtype=complex) / np.sqrt(2),
        "-":  np.array([1, -1], dtype=complex) / np.sqrt(2),
        "+i": np.array([1, 1j], dtype=complex) / np.sqrt(2),
        "-i": np.array([1, -1j], dtype=complex) / np.sqrt(2),
    }
    if s in named:
        return named[s]

    if "+" in s:
        parts = [p for p in s.split("+") if p != ""]
        if len(parts) == 0:
            raise ValueError("Invalid superposition string.")
        if not all(_is_binary_string(p) for p in parts):
            raise ValueError("Superposition must be like '0001+0010+1111' (binary terms only).")
        n = len(parts[0])
        if any(len(p) != n for p in parts):
            raise ValueError("All basis strings in the superposition must have the same length.")
        dim = 2**n
        vec = np.zeros(dim, dtype=complex)
        for p in parts:
            vec[int(p, 2)] += 1.0
        return _normalize_state(vec)

    if _is_binary_string(s):
        n = len(s)
        vec = np.zeros(2**n, dtype=complex)
        vec[int(s, 2)] = 1.0
        return vec

    raise ValueError(f"Unknown initial_state format: {initial_state}")

def infer_num_qubits_from_state(statevec: np.ndarray) -> int:
    dim = len(statevec)
    n = int(np.log2(dim))
    if 2**n != dim:
        raise ValueError("Statevector length must be a power of 2.")
    return n

def is_pauli_string(u: str) -> bool:
    if not isinstance(u, str):
        return False
    s = u.strip().upper()
    return len(s) > 0 and all(c in "IXYZ" for c in s)

def build_unitary_descriptor(unitary: Union[str, QuantumCircuit, np.ndarray, Callable]):
    """
    Returns one of:
      {"type":"pauli", "pauli": "XXIZ"}
      {"type":"gate", "gate": <Instruction>}
      {"type":"callable", "fn": callable}
    """
    if callable(unitary) and not isinstance(unitary, str):
        return {"type": "callable", "fn": unitary}

    if isinstance(unitary, str):
        s = unitary.strip().upper()
        if is_pauli_string(s):
            return {"type": "pauli", "pauli": s}
        raise ValueError(f"Unknown unitary string. Use Pauli strings like 'XXII': {unitary}")

    if isinstance(unitary, QuantumCircuit):
        return {"type": "gate", "gate": unitary.to_gate()}

    if isinstance(unitary, np.ndarray):
        mat = np.asarray(unitary, dtype=complex)
        return {"type": "gate", "gate": Operator(mat).to_instruction()}

    # Instruction / Gate objects also work with Operator; but keep it simple:
    try:
        op = Operator(unitary)
        return {"type": "gate", "gate": op.to_instruction()}
    except Exception:
        pass

    raise ValueError(f"Unsupported unitary type: {type(unitary)}")

def apply_branch_phase_on_controls(qc: QuantumCircuit, controls: List, phi: float):
    """
    Multiply amplitude of the branch where ALL controls are |1> by e^{i phi}.
    Uses Rz instead of P gate; the global phase difference does not affect
    measurement outcomes.
    """
    if abs(phi) < 1e-12:
        return
    if len(controls) == 1:
        qc.rz(phi, controls[0])
    else:
        # Decompose multi-controlled phase into Rz + CX ladder
        n = len(controls)
        target = controls[-1]
        qc.rz(phi / (2 ** (n - 1)), target)
        for i in range(n - 1):
            qc.cx(controls[i], target)
            qc.rz(-phi / (2 ** (n - 1 - i)), target)
            qc.cx(controls[i], target)
            if i < n - 2:
                qc.rz(phi / (2 ** (n - 2 - i)), target)

def apply_controlled_pauli_string(
    qc: QuantumCircuit,
    controls: List,
    targets_q0_to_qn1: List,
    pauli_msb_to_lsb: str,
):
    """
    Apply tensor-product Pauli given by a string of length n_target.
    Convention:
      pauli[0] acts on q_{n-1}, pauli[-1] acts on q_0.
    Here `targets_q0_to_qn1` is [q0, q1, ..., q_{n-1}] in register order.
    """
    pauli = pauli_msb_to_lsb.upper()
    n = len(targets_q0_to_qn1)
    if len(pauli) != n:
        raise ValueError(f"Pauli string length {len(pauli)} must equal number of target qubits {n}.")

    # map: char at rightmost -> q0, next -> q1, ... leftmost -> q_{n-1}
    pauli_lsb_to_msb = pauli[::-1]

    for qi, tq in enumerate(targets_q0_to_qn1):
        p = pauli_lsb_to_msb[qi]
        if p == "I":
            continue
        if p == "X":
            if len(controls) == 1:
                qc.cx(controls[0], tq)
            else:
                qc.mcx(controls, tq)
        elif p == "Z":
            # Controlled-Z = H on target, then CX/MCX, then H on target
            qc.h(tq)
            if len(controls) == 1:
                qc.cx(controls[0], tq)
            else:
                qc.mcx(controls, tq)
            qc.h(tq)
        elif p == "Y":
            # Y = S X S^\dagger  -> controlled-Y = S ; controlled-X ; S^\dagger
            qc.s(tq)
            if len(controls) == 1:
                qc.cx(controls[0], tq)
            else:
                qc.mcx(controls, tq)
            qc.sdg(tq)
        else:
            raise ValueError(f"Invalid Pauli char: {p}")


# -----------------------------------------------------------------------------
# LCU (postselected)
# -----------------------------------------------------------------------------

class LCU:
    def __init__(
        self,
        coefficients: List[Union[float, complex]],
        unitaries: List[Union[str, QuantumCircuit, np.ndarray, Callable]],
        initial_state: Union[str, List, np.ndarray],
        normalize_coeffs_l2: bool = True,
    ):
        if len(coefficients) != len(unitaries):
            raise ValueError("coefficients and unitaries must have the same length.")
        self.N = len(coefficients)
        if self.N < 1:
            raise ValueError("Need at least one term.")

        # initial state
        self.psi = parse_initial_state(initial_state)
        self.n_target = infer_num_qubits_from_state(self.psi)

        # coefficients (complex allowed), L2 normalize as requested
        coeffs = np.array(coefficients, dtype=complex)
        if normalize_coeffs_l2:
            coeffs = coeffs / np.linalg.norm(coeffs)
        self.coeffs = coeffs
        self.coeff_abs = np.abs(coeffs)
        self.coeff_phase = np.angle(coeffs)

        self.alpha_sum = float(np.sum(self.coeff_abs))
        if self.alpha_sum < 1e-15:
            raise ValueError("Sum of |coefficients| too small.")

        # ancilla count
        self.n_ancilla = int(np.ceil(np.log2(self.N))) if self.N > 1 else 1

        # ancilla amplitudes: sqrt(|a_i| / alpha)
        dim_a = 2**self.n_ancilla
        amps = np.zeros(dim_a, dtype=complex)
        amps[:self.N] = np.sqrt(self.coeff_abs / self.alpha_sum)
        self.ancilla_amps = _normalize_state(amps)

        # unitary descriptors
        self.unitaries = [build_unitary_descriptor(u) for u in unitaries]
        for d in self.unitaries:
            if d["type"] == "pauli" and len(d["pauli"]) != self.n_target:
                raise ValueError(
                    f"Pauli string {d['pauli']} has length {len(d['pauli'])}, "
                    f"but target has {self.n_target} qubits."
                )

    def build_circuit(self, measure_ancilla: bool = True) -> QuantumCircuit:
        anc = QuantumRegister(self.n_ancilla, "anc")
        tgt = QuantumRegister(self.n_target, "tgt")
        if measure_ancilla:
            c_anc = ClassicalRegister(self.n_ancilla, "c_anc")
            qc = QuantumCircuit(anc, tgt, c_anc)
        else:
            qc = QuantumCircuit(anc, tgt)

        # init target |psi>
        qc.initialize(self.psi, tgt)
        qc.barrier(label="|psi>")

        # PREPARE ancillas
        prep = StatePreparation(self.ancilla_amps)
        qc.append(prep, anc)
        qc.barrier(label="PREPARE")

        # Use MSB-first ancilla ordering for matching format(i,'0..b')
        anc_msb = list(anc)[::-1]             # [anc_{n-1}, ..., anc_0]
        tgt_q0_to_qn1 = list(tgt)             # [tgt_0, tgt_1, ..., tgt_{n-1}]

        # SELECT
        for i in range(self.N):
            bits = format(i, f"0{self.n_ancilla}b")  # MSB..LSB

            # flip 0-bits so the selected basis maps to all-ones
            for j, b in enumerate(bits):
                if b == "0":
                    qc.x(anc_msb[j])

            # coefficient phase e^{i arg(a_i)} on the selected branch
            apply_branch_phase_on_controls(qc, anc_msb, float(self.coeff_phase[i]))

            desc = self.unitaries[i]
            if desc["type"] == "pauli":
                apply_controlled_pauli_string(qc, anc_msb, tgt_q0_to_qn1, desc["pauli"])
            elif desc["type"] == "gate":
                g = desc["gate"]
                if g.num_qubits != self.n_target:
                    raise ValueError(
                        f"Gate acts on {g.num_qubits} qubits, expected {self.n_target}."
                    )
                qc.append(g.control(self.n_ancilla), anc_msb + tgt_q0_to_qn1)
            else:
                # callable signature: fn(qc, target_qubits_q0_to_qn1, control_qubits_msb_to_lsb)
                desc["fn"](qc, tgt_q0_to_qn1, anc_msb)

            # undo flips
            for j, b in enumerate(bits):
                if b == "0":
                    qc.x(anc_msb[j])

        qc.barrier(label="SELECT")

        # UNPREPARE = PREPARE†
        qc.append(prep.inverse(), anc)
        qc.barrier(label="PREPARE†")

        if measure_ancilla:
            qc.measure(anc, qc.cregs[0])

        return qc

    def pauli_string_matrix(self, pauli_msb_to_lsb: str) -> np.ndarray:
        pauli = pauli_msb_to_lsb.upper()
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        m = {"I": I, "X": X, "Y": Y, "Z": Z}

        U = np.array([[1]], dtype=complex)
        # kron left->right corresponds to acting on q_{n-1} ... q_0 (matches our bitstring convention)
        for p in pauli:
            U = np.kron(U, m[p])
        return U

    def theoretical_success_probability(self) -> float:
        """
        Exact:
          P = || A|psi> ||^2 / (sum_i |a_i|)^2
        with A = sum_i a_i U_i
        Only available when each unitary is a Pauli string or Gate/Circuit/Matrix.
        """
        dim = 2**self.n_target
        psi = self.psi.reshape(dim, 1)
        Apsi = np.zeros((dim, 1), dtype=complex)

        for i in range(self.N):
            a = self.coeffs[i]
            desc = self.unitaries[i]
            if desc["type"] == "pauli":
                U = self.pauli_string_matrix(desc["pauli"])
            elif desc["type"] == "gate":
                U = Operator(desc["gate"]).data
            else:
                raise ValueError("Theory requires matrices/circuits for callable unitaries.")
            Apsi += a * (U @ psi)

        p = float((np.vdot(Apsi, Apsi).real) / (self.alpha_sum**2))
        return max(0.0, min(1.0, p))

    def run(
        self,
        shots: int = 8192,
        visualize: bool = True,
        seed: Optional[int] = 7,
        show_circuit: bool = False,
    ):
        qc = self.build_circuit(measure_ancilla=True)

        sim = AerSimulator(seed_simulator=seed)
        tqc = transpile(qc, sim, basis_gates=[
            'rx', 'ry', 'rz', 'x', 'cx', 'ccx', 'mcx', 's', 'sdg', 'h', 'measure',
        ])
        result = sim.run(tqc, shots=shots).result()
        counts = result.get_counts()

        success_key = "0" * self.n_ancilla
        p_emp = counts.get(success_key, 0) / shots

        p_theory = self.theoretical_success_probability()

        print("=" * 80)
        print("LCU RUN")
        print("=" * 80)
        print(f"N terms:                  {self.N}")
        print(f"Target qubits:            {self.n_target}")
        print(f"Ancilla qubits:           {self.n_ancilla}")
        print(f"Shots:                    {shots}")
        print(f"Success key:              {success_key}")
        print("-" * 80)
        print(f"Success prob (empirical): {p_emp:.6f}  ({100*p_emp:.2f}%)")
        print(f"Success prob (theory):    {p_theory:.6f}  ({100*p_theory:.2f}%)")
        print("-" * 80)
        print(f"Circuit depth:            {qc.depth()}")
        print(f"Total gates:              {qc.size()}")
        print("=" * 80)

        if visualize:
            # --- Histogram of measurement outcomes ---
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_histogram(counts, ax=ax)
            plt.tight_layout()
            plt.show()

            # --- Success probability comparison plot ---
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            labels = ['Empirical', 'Theoretical']
            values = [p_emp, p_theory]
            colors = ['#1565C0', '#00897B']
            bars = ax2.bar(labels, values, color=colors, width=0.5, edgecolor='#333333', linewidth=0.8)
            for bar, val in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                         f'{val:.4f}\n({100*val:.2f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax2.set_ylabel('Success Probability', fontsize=12)
            ax2.set_title('LCU Postselection Success Probability', fontsize=13, fontweight='bold')
            ax2.set_ylim(0, max(values) * 1.25)
            ax2.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            fig2.savefig('success_probability.png', dpi=150, bbox_inches='tight')
            print("Success probability plot saved to success_probability.png")
            plt.show()

        if show_circuit:
            style = {
                'fontsize': 10,
                'subfontsize': 8,
                'displaycolor': {
                    'X': ('#1565C0', '#FFFFFF'),
                    'cx': ('#1565C0', '#FFFFFF'),
                    'mcx': ('#0D47A1', '#FFFFFF'),
                    'S': ('#00897B', '#FFFFFF'),
                    'Sdg': ('#00897B', '#FFFFFF'),
                    'H': ('#7B1FA2', '#FFFFFF'),
                    'Rx': ('#E65100', '#FFFFFF'),
                    'Ry': ('#E65100', '#FFFFFF'),
                    'Rz': ('#E65100', '#FFFFFF'),
                },
                'creglinestyle': 'solid',
                'dpi': 150,
            }
            fig = qc.draw(
                output='mpl',
                fold=-1,
                scale=0.6,
                style=style,
            )
            fig.savefig('lcu_circuit.png', dpi=150, bbox_inches='tight')
            print("Circuit diagram saved to lcu_circuit.png")
            plt.show()

        return {
            "circuit": qc,
            "counts": counts,
            "shots": shots,
            "success_key": success_key,
            "p_success_emp": p_emp,
            "p_success_theory": p_theory,
        }


# -----------------------------------------------------------------------------
# Section for different tests
# -----------------------------------------------------------------------------

def main():
    # Term:
    # 0.3*XXXXY + 0.89*XYZZY + 0.99*YYXXI + 0.88*YXYXZ + 0.98*XYYZI

    coefficients = [0.3, 0.89, 0.99, 0.88, 0.98]
    unitaries    = ["XXXXY", "XYZZY", "YYXXI", "YXYXZ", "XYYZI"]
    initial_state = "00000+10101+00100+11111"

    lcu = LCU(coefficients, unitaries, initial_state, normalize_coeffs_l2=True)
    lcu.run(shots=20000, visualize=True, show_circuit=True)

if __name__ == "__main__":
    main()