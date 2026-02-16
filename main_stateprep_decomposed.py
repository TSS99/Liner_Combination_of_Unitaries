# =============================================================================
# LCU (postselected)
# StatePreparation and StatePreparation† are inlined and decomposed to ONLY:
#   - ry
#   - cx
#
# Important limitation:
#   Decomposition to {ry, cx} only is generally possible for REAL statevectors
#   (no relative complex phases). If you pass a complex initial_state like "+i",
#   this code will raise an error.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from typing import List, Union, Callable, Optional

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator, Statevector


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

def _assert_real_state(vec: np.ndarray, name: str, tol: float = 1e-10):
    vec = np.asarray(vec, dtype=complex)
    if np.max(np.abs(np.imag(vec))) > tol:
        raise ValueError(
            f"{name} has complex phases. Pure {{ry, cx}} state-prep is not generally possible. "
            f"Use a real-amplitude state (no relative complex phases), or allow rz as well."
        )

def parse_initial_state(initial_state: Union[str, List, np.ndarray]) -> np.ndarray:
    """
    Supported:
      - list/np.array statevector
      - '|0>', '|1>', '|+>', '|->', '|+i>', '|-i>'
      - '0101' (basis)
      - '0001+0010+1111' (equal superposition)
      - '0+1' (1-qubit equal superposition of |0> and |1>)
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

    try:
        op = Operator(unitary)
        return {"type": "gate", "gate": op.to_instruction()}
    except Exception:
        pass

    raise ValueError(f"Unsupported unitary type: {type(unitary)}")

def compile_stateprep_ry_cx(statevec: np.ndarray, n_qubits: int, name: str, tol: float = 1e-10) -> QuantumCircuit:
    """
    Build a circuit that prepares `statevec` from |0..0> using StatePreparation,
    then transpile it down to ONLY {ry, cx}. The resulting circuit is inlined
    into the main circuit so you actually see the decomposition.

    Works reliably for real-amplitude states. Throws if state has nontrivial phases.
    """
    vec = _normalize_state(statevec)
    _assert_real_state(vec, name=name, tol=tol)

    c = QuantumCircuit(n_qubits, name=name)
    c.append(StatePreparation(vec), list(range(n_qubits)))

    # Try optimization level 3 to get cleanest circuit
    tc = transpile(c, basis_gates=["ry", "cx", "rz", "u", "h", "x"], optimization_level=3)
    
    # We build a 'candidate' circuit with only {ry, cx}.
    # We iterate forward.
    # For Rz/U phases: Try to snap to 0 or pi.
    
    cleaned = QuantumCircuit(n_qubits, name=name)
    z_pending = [False] * n_qubits

    for inst in tc.data:
        op = inst.operation
        qargs = inst.qubits
        q_idx = [tc.find_bit(q).index for q in qargs]
        
        opname = op.name
        
        if opname == "id" or opname == "barrier":
            continue
            
        elif opname == "cx":
            # Z on control passes through. Z on target becomes Z_target * Z_control.
            c_idx, t_idx = q_idx
            if z_pending[t_idx]:
                z_pending[c_idx] = not z_pending[c_idx]
                # z_pending[t_idx] remains True
            cleaned.cx(c_idx, t_idx)
            
        elif opname == "ry":
            q = q_idx[0]
            theta = float(op.params[0])
            if z_pending[q]:
                theta = -theta
            cleaned.ry(theta, q)
            
        elif opname == "x":
            q = q_idx[0]
            # X = Ry(pi) Z (approx). 
            theta = np.pi
            if z_pending[q]:
                theta = -theta
            cleaned.ry(theta, q)
            # Z X = -X Z. Z passes through.
            
        elif opname == "h":
            q = q_idx[0]
            # H = Ry(pi/2) * Z.
            theta = np.pi/2
            if z_pending[q]:
                theta = -theta
            cleaned.ry(theta, q)
            z_pending[q] = not z_pending[q]
            
        elif opname == "rz":
            q = q_idx[0]
            phi = float(op.params[0])
            # Snap to grid
            steps = round(phi / np.pi)
            if steps % 2 == 1:
                z_pending[q] = not z_pending[q]
                
        elif opname == "u":
            q = q_idx[0]
            theta, phi, lam = map(float, op.params)
            
            # 1. Rz(lam)
            steps_lam = round(lam / np.pi)
            if steps_lam % 2 == 1:
                z_pending[q] = not z_pending[q]
                
            # 2. Ry(theta)
            if z_pending[q]:
                theta = -theta
            cleaned.ry(theta, q)
            
            # 3. Rz(phi)
            steps_phi = round(phi / np.pi)
            if steps_phi % 2 == 1:
                z_pending[q] = not z_pending[q]
    
    # Verification (Optional debug)
    # psi_target = Statevector(vec)
    # psi_cleaned = Statevector.from_instruction(cleaned)
    # overlap = abs(psi_target.inner(psi_cleaned))**2
    # if overlap < 0.9:
    #    print(f"[{name}] Warning: Overlap {overlap:.4f} < 0.9. Z-pending at end: {z_pending}")

    return cleaned

def apply_branch_phase_on_controls(qc: QuantumCircuit, controls: List, phi: float):
    """
    Multiply amplitude of the branch where ALL controls are |1> by exp(i*phi).
    This is used for coefficient phases during SELECT.
    """
    if abs(phi) < 1e-12:
        return
    if len(controls) == 1:
        qc.p(phi, controls[0])
        return

    ctrl = controls[:-1]
    tgt = controls[-1]

    if hasattr(qc, "mcp"):
        qc.mcp(phi, ctrl, tgt)
    else:
        from qiskit.circuit.library import PhaseGate
        qc.append(PhaseGate(phi).control(len(ctrl)), ctrl + [tgt])

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
            qc.h(tq)
            if len(controls) == 1:
                qc.cx(controls[0], tq)
            else:
                qc.mcx(controls, tq)
            qc.h(tq)
        elif p == "Y":
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

        self.psi = parse_initial_state(initial_state)
        self.n_target = infer_num_qubits_from_state(self.psi)

        coeffs = np.array(coefficients, dtype=complex)
        if normalize_coeffs_l2:
            coeffs = coeffs / np.linalg.norm(coeffs)
        self.coeffs = coeffs
        self.coeff_abs = np.abs(coeffs)
        self.coeff_phase = np.angle(coeffs)

        self.alpha_sum = float(np.sum(self.coeff_abs))
        if self.alpha_sum < 1e-15:
            raise ValueError("Sum of |coefficients| too small.")

        self.n_ancilla = int(np.ceil(np.log2(self.N))) if self.N > 1 else 1

        dim_a = 2**self.n_ancilla
        amps = np.zeros(dim_a, dtype=complex)
        amps[:self.N] = np.sqrt(self.coeff_abs / self.alpha_sum)
        self.ancilla_amps = _normalize_state(amps)

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

        # -----------------------
        # Target state prep (RY + CX only)
        # -----------------------
        tgt_prep = compile_stateprep_ry_cx(self.psi, self.n_target, name="TARGET_PREP_RY_CX")
        qc.compose(tgt_prep, qubits=list(tgt), inplace=True)
        qc.barrier(label="|psi>")

        # -----------------------
        # Ancilla PREPARE (RY + CX only)
        # -----------------------
        anc_prep = compile_stateprep_ry_cx(self.ancilla_amps, self.n_ancilla, name="PREPARE_RY_CX")
        qc.compose(anc_prep, qubits=list(anc), inplace=True)
        qc.barrier(label="PREPARE")

        # MSB-first ordering for SELECT logic
        anc_msb = list(anc)[::-1]
        tgt_q0_to_qn1 = list(tgt)

        # SELECT
        for i in range(self.N):
            bits = format(i, f"0{self.n_ancilla}b")  # MSB..LSB

            for j, b in enumerate(bits):
                if b == "0":
                    qc.x(anc_msb[j])

            apply_branch_phase_on_controls(qc, anc_msb, float(self.coeff_phase[i]))

            desc = self.unitaries[i]
            if desc["type"] == "pauli":
                apply_controlled_pauli_string(qc, anc_msb, tgt_q0_to_qn1, desc["pauli"])
            elif desc["type"] == "gate":
                g = desc["gate"]
                if g.num_qubits != self.n_target:
                    raise ValueError(f"Gate acts on {g.num_qubits} qubits, expected {self.n_target}.")
                qc.append(g.control(self.n_ancilla), anc_msb + tgt_q0_to_qn1)
            else:
                desc["fn"](qc, tgt_q0_to_qn1, anc_msb)

            for j, b in enumerate(bits):
                if b == "0":
                    qc.x(anc_msb[j])

        qc.barrier(label="SELECT")

        # -----------------------
        # Ancilla UNPREPARE (PREPARE†) (still RY + CX only)
        # -----------------------
        qc.compose(anc_prep.inverse(), qubits=list(anc), inplace=True)
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
        for p in pauli:
            U = np.kron(U, m[p])
        return U

    def theoretical_success_probability(self) -> float:
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
        tqc = transpile(
            qc,
            sim,
            basis_gates=["ry", "cx", "x", "h", "s", "sdg", "rz", "p", "ccx", "mcx", "measure"],
            optimization_level=0,
            seed_transpiler=seed,
        )

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
        print(f"Circuit depth (original): {qc.depth()}")
        print(f"Total gates (original):   {qc.size()}")
        print(f"Depth (transpiled):       {tqc.depth()}")
        print(f"Gates (transpiled):       {tqc.size()}")
        print("=" * 80)

        if visualize:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_histogram(counts, ax=ax)
            plt.tight_layout()
            plt.show()

        if show_circuit:
            fig = qc.draw(output="mpl", fold=-1, scale=0.6)
            fig.savefig("lcu_circuit_with_rycx_stateprep.png", dpi=150, bbox_inches="tight")
            print("Circuit diagram saved to lcu_circuit_with_rycx_stateprep.png")
            plt.show()

        return {
            "circuit": qc,
            "circuit_transpiled": tqc,
            "counts": counts,
            "shots": shots,
            "success_key": success_key,
            "p_success_emp": p_emp,
            "p_success_theory": p_theory,
        }


# -----------------------------------------------------------------------------
# For different kinds of tests
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

