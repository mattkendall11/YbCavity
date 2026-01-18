"""
Tavis-Cummings (Tavis–Cummings) QuTiP framework for Ytterbium in a cavity
----------------------------------------------------------------------

Notes / modelling choices:
- Atoms are modelled as two-level systems (|g>,|e>) by default. For Yb
  you'll typically encode the logical qubit in long-lived low-lying states
  and use a Rydberg/excited level for cavity coupling. The TC Hamiltonian
  below assumes a single (relevant) transition per atom couples to the
  cavity mode.
- The code is purposely modular: you can replace two-level atoms with
  higher-dimensional atoms, or add an explicit Rydberg level later.
- Quantities are SI where relevant (Hz, s^-1, meters, coulomb·m). The
  user must supply physically consistent parameters.

Dependencies:
  - qutip
  - numpy
  - scipy

Recommended usage:
  - Import functions from this file into a Jupyter / script environment.
  - Start with small Hilbert-space truncation for the cavity (n_cav=6..20)
    and increase until results converge.

"""

import numpy as np
from math import pi
from scipy import constants as const

# QuTiP imports
from qutip import (
    tensor, qeye, basis, destroy, sigmax, sigmay, sigmaz, sigmap, sigmam,
    mesolve, propagator, expect, fidelity
)

# -------------------------------------------------------------------------
# Basic operator constructors
# -------------------------------------------------------------------------

def _single_atom_operators():
    """Return the standard 2x2 operators for a single two-level atom.

    Returns
    -------
    sm, sp, sz, id2
        Lowering, raising, Pauli-z, and identity operators for a single
        two-level system in the computational basis |g>=(1,0), |e>=(0,1).
    """
    sm = sigmam()
    sp = sigmap()
    sz = sigmaz()
    id2 = qeye(2)
    return sm, sp, sz, id2


def tensor_identity_list(local_op, position, N):
    """Embed a local operator into the full N-atom Hilbert space.

    Parameters
    ----------
    local_op : Qobj
        Operator acting on a single atom (2x2)
    position : int
        Index of the atom the operator acts on (0-indexed)
    N : int
        Total number of atoms

    Returns
    -------
    Qobj
        Tensor product with identities on other atom spaces.
    """
    op_list = [qeye(2) for _ in range(N)]
    op_list[position] = local_op
    return tensor(op_list)


# -------------------------------------------------------------------------
# Tavis-Cummings Hamiltonian builder
# -------------------------------------------------------------------------

def build_tavis_cummings_H(N, g_list, w_c, w_a_list, n_cav=10, hbar=1.0):
    """Construct the full Tavis–Cummings Hamiltonian for N two-level atoms
    coupled to a single cavity mode.

    H = hbar * ( w_c a^\dag a + sum_j (w_a_j/2) sigma_j^z )
        + hbar * sum_j g_j ( a^\dag sigma_j^- + a sigma_j^+ )

    Parameters
    ----------
    N : int
        Number of atoms (two-level systems)
    g_list : array_like, shape (N,)
        Coupling strengths g_j in Hz (angular frequency units: rad/s).
        If a scalar is supplied, it is broadcast to all atoms.
    w_c : float
        Cavity frequency in rad/s (omega_c)
    w_a_list : array_like, shape (N,)
        Atomic transition frequencies (omega_a_j) in rad/s.
    n_cav : int
        Truncation dimension for the cavity Hilbert space (cavity Fock)
    hbar : float
        Planck's constant over 2pi. Defaults to 1. Use const.hbar to work in SI.

    Returns
    -------
    Qobj
        Hamiltonian as a QuTiP Qobj acting on Hilbert space (cavity) x (atoms)
    """
    # Cast g_list and w_a_list to arrays
    g_arr = np.array(g_list, ndmin=1)
    if g_arr.size == 1:
        g_arr = np.ones(N) * float(g_arr)
    w_a_arr = np.array(w_a_list, ndmin=1)
    if w_a_arr.size == 1:
        w_a_arr = np.ones(N) * float(w_a_arr)

    # Cavity operators
    a = destroy(n_cav)
    id_c = qeye(n_cav)

    # Atomic local operators (in 2-level space)
    sm, sp, sz, id2 = _single_atom_operators()

    # Build atomic operators in full tensor space (atoms only)
    # We'll build tensor product between cavity and atomic tensor space later
    atom_sm_list = [tensor_identity_list(sm, j, N) for j in range(N)]
    atom_sp_list = [tensor_identity_list(sp, j, N) for j in range(N)]
    atom_sz_list = [tensor_identity_list(sz, j, N) for j in range(N)]
    id_atom = tensor([qeye(2) for _ in range(N)]) if N > 0 else qeye(1)

    # Promote cavity operators into full space: tensor(cavity, atoms...)
    a_full = tensor(a, id_atom)
    adag_full = a_full.dag()

    # Bare Hamiltonians
    H_cav = hbar * w_c * adag_full * a_full

    # Atomic energy: (w_a/2) sigma^z  per atom
    H_atoms = 0
    for j in range(N):
        H_atoms += hbar * (w_a_arr[j] / 2.0) * tensor(qeye(n_cav), atom_sz_list[j])

    # Interaction term
    H_int = 0
    for j in range(N):
        g_j = float(g_arr[j])
        # promote sm/sp to include cavity identity
        sm_full = tensor(qeye(n_cav), atom_sm_list[j])
        sp_full = tensor(qeye(n_cav), atom_sp_list[j])
        H_int += hbar * g_j * (adag_full * sm_full + a_full * sp_full)

    H = H_cav + H_atoms + H_int
    return H


# -------------------------------------------------------------------------
# Collapse operators (Lindblad) builder
# -------------------------------------------------------------------------

def build_collapse_operators(N, n_cav, kappa=0.0, gamma_list=None, gamma_phi_list=None):
    """Return a list of QuTiP collapse operators for the cavity+N atoms system.

    - cavity loss: sqrt(kappa) * a
    - atom relaxation: sqrt(gamma_j) * sigma_j^-
    - atom dephasing: sqrt(gamma_phi_j) * sigma_j^z

    Parameters
    ----------
    N : int
        Number of atoms
    n_cav : int
        Cavity truncation used when building the Hamiltonian
    kappa : float
        Cavity energy decay rate (Hz)
    gamma_list : array_like or None
        List of atomic population decay rates gamma_j in Hz. If None, zeros are used.
    gamma_phi_list : array_like or None
        List of atomic pure-dephasing rates in Hz. If None, zeros are used.

    Returns
    -------
    list
        List of QuTiP collapse operators on the full Hilbert space.
    """
    # Prepare lists
    if gamma_list is None:
        gamma_list = np.zeros(N)
    if gamma_phi_list is None:
        gamma_phi_list = np.zeros(N)

    # Cavity operator
    a = destroy(n_cav)
    id_atom = tensor([qeye(2) for _ in range(N)]) if N > 0 else qeye(1)
    a_full = tensor(a, id_atom)

    collapses = []
    if kappa > 0.0:
        collapses.append(np.sqrt(kappa) * a_full)

    # Atomic collapses
    sm, sp, sz, id2 = _single_atom_operators()
    for j in range(N):
        # lowering
        c_relax = np.sqrt(float(gamma_list[j])) * tensor(qeye(n_cav), tensor_identity_list(sm, j, N))
        if float(gamma_list[j]) > 0.0:
            collapses.append(c_relax)
        # pure dephasing (L ~ sigma_z)
        if float(gamma_phi_list[j]) > 0.0:
            c_deph = np.sqrt(float(gamma_phi_list[j])) * tensor(qeye(n_cav), tensor_identity_list(sz, j, N))
            collapses.append(c_deph)

    return collapses


# -------------------------------------------------------------------------
# Cooperativity utilities
# -------------------------------------------------------------------------

def cooperativity_single(g, kappa, gamma):
    """Return single-atom cooperativity C = 4 g^2 / (kappa gamma).

    All arguments must be in angular-frequency units (rad/s).
    """
    return 4.0 * (g ** 2) / (kappa * gamma)


def cooperativity_collective(C_single, N):
    """Collective cooperativity C_N = N * C_single for uniform coupling."""
    return N * C_single


# -------------------------------------------------------------------------
# Dispersive effective Hamiltonian (second-order Schrieffer–Wolff approx)
# -------------------------------------------------------------------------

def dispersive_effective_H(N, g_list, Delta_list, n_cav, hbar=1.0):
    """Construct a second-order dispersive effective Hamiltonian for N atoms
    after eliminating the cavity mode to leading order in g/Delta.

    H_eff ~= sum_j (g_j^2 / Delta_j) sigma_j^+ sigma_j^- + sum_{i!=j} (g_i g_j / Delta_avg) (sigma_i^+ sigma_j^- + h.c.)

    WARNING: This is an approximate Hamiltonian that neglects higher-order
    terms and frequency shifts. Use it for intuition and quick estimates, but
    be cautious for quantitative predictions near resonance or large g/Delta.
    """
    # Normalize lists
    g_arr = np.array(g_list, ndmin=1)
    if g_arr.size == 1:
        g_arr = np.ones(N) * float(g_arr)
    Delta_arr = np.array(Delta_list, ndmin=1)
    if Delta_arr.size == 1:
        Delta_arr = np.ones(N) * float(Delta_arr)

    # atomic operators in atoms-only tensor space
    sm, sp, sz, id2 = _single_atom_operators()
    atom_sm_list = [tensor_identity_list(sm, j, N) for j in range(N)]
    atom_sp_list = [tensor_identity_list(sp, j, N) for j in range(N)]

    H_atoms_eff = 0
    # On-site ac-Stark / Lamb shift like terms
    for j in range(N):
        H_atoms_eff += (g_arr[j] ** 2 / Delta_arr[j]) * atom_sp_list[j] * atom_sm_list[j]

    # Exchange terms (i != j)
    H_exch = 0
    # define a sensible average Delta for cross-terms
    Delta_avg = np.mean(Delta_arr)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            H_exch += (g_arr[i] * g_arr[j] / Delta_avg) * (atom_sp_list[i] * atom_sm_list[j] + atom_sm_list[i] * atom_sp_list[j])

    # Promote to full Hilbert space by tensoring with cavity vacuum projector implicitly
    # (we return an atoms-only Hamiltonian — this is convenient for some analyses)
    H_eff = H_atoms_eff + H_exch
    return H_eff


# -------------------------------------------------------------------------
# Simple simulation driver & fidelity utilities
# -------------------------------------------------------------------------

def propagate_and_compute_state_fidelities(H, collapses, rho0, tlist):
    """Propagate a density matrix and return state fidelities vs. time.

    This is a thin wrapper over qutip.mesolve to produce time-series results.

    Parameters
    ----------
    H : Qobj
        Hamiltonian (full Hilbert space)
    collapses : list
        List of collapse operators
    rho0 : Qobj
        Initial density matrix (or state vector)
    tlist : array_like
        Times to sample

    Returns
    -------
    result
        qutip.Result object with states, expect, etc.
    """
    result = mesolve(H, rho0, tlist, c_ops=collapses, e_ops=[])
    return result


def compute_average_gate_fidelity_unitary_with_noise(U_target, H, collapses, psi_basis, t_gate):
    """Estimate average gate fidelity for a target unitary U_target by evolving
    computational basis states under the noisy dynamics and averaging final
    state fidelities.

    - psi_basis : list of state vectors for computational basis (Qobj kets)
    - U_target : target unitary (Qobj) acting on the same Hilbert space
    - H, collapses, t_gate : dynamics to simulate

    NOTE: This is a simple (state-based) average fidelity estimate. It does not
    reconstruct a full process matrix. For two-qubit gates this is a good
    first-order diagnostic. For full process fidelity use process tomography
    routines (not included here for brevity).
    """
    # Ensure t_gate is scalar or small list; we compute final states at t_gate
    if np.isscalar(t_gate):
        tlist = [0.0, t_gate]
    else:
        tlist = [0.0, t_gate]

    fidelities = []
    for psi in psi_basis:
        # make density matrix
        rho0 = psi.proj()
        out = mesolve(H, rho0, tlist, c_ops=collapses, e_ops=[])
        rho_final = out.states[-1]
        # ideal final state
        psi_ideal = (U_target * psi).unit()
        fid = fidelity(rho_final, psi_ideal.proj())
        fidelities.append(fid)

    return np.mean(fidelities), fidelities


# -------------------------------------------------------------------------
# Example parameter block (placeholder values) & example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: two atoms (N=2), small cavity truncation
    N = 2
    n_cav = 8

    # --- Placeholders (USER: replace with physically consistent values) ---
    # Frequencies in angular units (rad/s). Use 2*pi*freq_in_Hz to convert.
    omega_c = 2 * pi * 10e9      # cavity at 10 GHz (angular)
    omega_a = 2 * pi * 10e9 + 2 * pi * 10e6  # atom detuned by 10 MHz

    # Coupling strengths g_j (rad/s). Placeholder; compute from dipole and V.
    g0 = 2 * pi * 1e5  # e.g., 100 kHz coupling
    g_list = [g0 for _ in range(N)]

    # Dissipation
    kappa = 2 * pi * 1e3   # cavity decay 1 kHz
    gamma_list = [2 * pi * 10.0 for _ in range(N)]  # atom decay 10 Hz
    gamma_phi_list = [2 * pi * 1.0 for _ in range(N)]  # dephasing 1 Hz

    # Build full Hamiltonian
    H = build_tavis_cummings_H(N, g_list, omega_c, [omega_a]*N, n_cav=n_cav, hbar=const.hbar)

    # Build collapse operators
    collapses = build_collapse_operators(N, n_cav, kappa=kappa, gamma_list=gamma_list, gamma_phi_list=gamma_phi_list)

    # Cooperativity single
    C1 = cooperativity_single(g0, kappa, gamma_list[0])
    print('Single-atom cooperativity (C):', C1)
    print('Collective cooperativity (C_N):', cooperativity_collective(C1, N))

    # Prepare computational basis states for two qubits (atoms only) embedded into full space
    # computational basis (|gg>, |ge>, |eg>, |ee>) tensored with cavity vacuum
    vac = basis(n_cav, 0)
    # atomic basis kets
    g = basis(2, 0)
    e = basis(2, 1)
    atom_basis = [tensor(vac, tensor(g, g)), tensor(vac, tensor(g, e)),
                  tensor(vac, tensor(e, g)), tensor(vac, tensor(e, e))]

    # A target unitary for demonstration: iSWAP generated by effective J = g^2/Delta
    Delta = omega_a - omega_c
    J = g0**2 / Delta
    t_iswap = pi / (2.0 * J)
    # Build an ideal iSWAP on the atomic subspace (not provided here) - user would
    # construct a 4x4 unitary and then promote to full space when needed.

    # Small propagation example: single basis state
    rho0 = atom_basis[1].proj()
    tlist = np.linspace(0.0, t_iswap, 101)
    result = mesolve(H, rho0, tlist, c_ops=collapses)
    print('Simulation done; final population (example):', expect(tensor(qeye(n_cav), tensor(sigmap()*sigmam(), qeye(2))), result.states[-1]))

    # Note: The example above is illustrative. Replace placeholder numbers with
    # realistic parameters for Yb Rydberg / transition dipoles and cavity geometry.

