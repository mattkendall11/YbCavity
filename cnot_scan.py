"""
scan_g_fidelity.py

Sweep single-atom coupling g, simulate noisy two-atom dynamics in a cavity,
compute average two-qubit gate fidelity for the native iSWAP gate, and plot
fidelity vs g and fidelity vs collective cooperativity C_N.

Requires: numpy, matplotlib, qutip
"""

import numpy as np
from tqdm import tqdm
from math import pi
import matplotlib.pyplot as plt
from qutip import (
    basis, tensor, qeye, destroy, sigmap, sigmam, sigmaz,
    mesolve, fidelity,Options
)
opts = Options(
    nsteps=200000000,      # default is ~1000 (too small)
    atol=1e-8,
    rtol=1e-6,
    method="bdf",      # IMPORTANT: stiff solver
    progress_bar=None
)

# -------------------------
# Helper / builder functions
# -------------------------
def build_tc_hamiltonian_two_atoms(g_vals, omega_c, omega_a, n_cav):
    """Return H for two atoms coupled with *identical* g (list or scalar).
    H uses ordering: (cavity) x (atom0 tensor atom1).
    All frequencies in rad/s; g in rad/s.
    """
    # cavity
    a = destroy(n_cav)
    id_c = qeye(n_cav)

    # two-level atomic operators (atoms-only)
    sm = sigmam()
    sp = sigmap()
    sz = sigmaz()
    id2 = qeye(2)

    # atoms-only operators embedded
    sm_0 = tensor(sm, id2)
    sm_1 = tensor(id2, sm)
    sp_0 = tensor(sp, id2)
    sp_1 = tensor(id2, sp)
    sz_0 = tensor(sz, id2)
    sz_1 = tensor(id2, sz)

    # promote atoms operators to full space (tensor with cavity identity)
    id_atoms = tensor(id2, id2)
    a_full = tensor(a, id_atoms)
    adag_full = a_full.dag()

    # # Hamiltonian pieces
    # H_cav = omega_c * adag_full * a_full  # hbar factor omitted (we use rad/s units)
    # H_atoms = 0.5 * omega_a * tensor(qeye(n_cav), sz_0) + 0.5 * omega_a * tensor(qeye(n_cav), sz_1)
    Delta = omega_a - omega_c
    H_cav = Delta * adag_full * a_full
    H_atoms = 0 * tensor(qeye(n_cav), sz_0)  # or drop entirely

    # assume identical g (scalar) or list/array length 2
    if np.isscalar(g_vals):
        g0 = float(g_vals)
        g1 = float(g_vals)
    else:
        g0 = float(g_vals[0]); g1 = float(g_vals[1])

    H_int = g0 * (adag_full * tensor(qeye(n_cav), sm_0) + a_full * tensor(qeye(n_cav), sp_0))
    H_int += g1 * (adag_full * tensor(qeye(n_cav), sm_1) + a_full * tensor(qeye(n_cav), sp_1))

    H = H_cav + H_atoms + H_int
    return H

def build_collapse_ops(n_cav, N=2, kappa=0.0, gamma=0.0, gamma_phi=0.0):
    """Return list of collapse operators for cavity + 2 atoms.
    gamma, gamma_phi are per-atom rates (rad/s)."""
    a = destroy(n_cav)
    id2 = qeye(2)
    id_atoms = tensor(id2, id2)

    a_full = tensor(a, id_atoms)
    c_ops = []
    if kappa > 0:
        c_ops.append(np.sqrt(kappa) * a_full)

    # atomic lowering operators embedded into full space
    sm = sigmam()
    sm_0 = tensor(qeye(n_cav), tensor(sm, id2))
    sm_1 = tensor(qeye(n_cav), tensor(id2, sm))
    if gamma > 0:
        c_ops.append(np.sqrt(gamma) * sm_0)
        c_ops.append(np.sqrt(gamma) * sm_1)
    if gamma_phi > 0:
        sz = sigmaz()
        sz_0 = tensor(qeye(n_cav), tensor(sz, id2))
        sz_1 = tensor(qeye(n_cav), tensor(id2, sz))
        # NOTE: convention: choose prefactor so that dephasing rate matches your T_phi
        c_ops.append(np.sqrt(gamma_phi) * sz_0)
        c_ops.append(np.sqrt(gamma_phi) * sz_1)

    return c_ops

# -------------------------
# Fidelity utilities
# -------------------------
def iSWAP_unitary_atom():
    """Return ideal iSWAP on the two-qubit atomic subspace (4x4 Qobj as numpy array)."""
    # Basis ordering: |00>, |01>, |10>, |11>
    # iSWAP = diag(1,---,---,1) with off-diagonals mapping 01 <-> 10 with i
    U = np.zeros((4,4), dtype=complex)
    U[0,0] = 1.0
    U[3,3] = 1.0
    U[1,2] = 1j
    U[2,1] = 1j
    return U

def atomic_basis_states_as_full(n_cav):
    """Return list of 4 kets (Qobj) for computational basis embedded with cavity vacuum."""
    vac = basis(n_cav, 0)
    g = basis(2, 0)
    e = basis(2, 1)
    # atom states in order |gg>,|ge>,|eg>,|ee> with cavity vacuum tensored first
    psi00 = tensor(vac, tensor(g, g))
    psi01 = tensor(vac, tensor(g, e))
    psi10 = tensor(vac, tensor(e, g))
    psi11 = tensor(vac, tensor(e, e))
    return [psi00, psi01, psi10, psi11]

def avg_gate_fidelity_iSWAP(H, c_ops, t_gate, n_cav):
    """Compute average state fidelity over computational basis for an iSWAP gate."""
    psi_basis = atomic_basis_states_as_full(n_cav)
    U_atom = iSWAP_unitary_atom()
    # embed U_atom into full Hilbert space (cavity identity x U_atom)
    # create Qobj for U_target using qutip, but simplest to multiply basis kets
    fid_list = []
    for psi in psi_basis:
        rho0 = psi.proj()
        # solve to t_gate
        out = mesolve(H, rho0, [0.0, t_gate], c_ops=c_ops,options = opts)
        rho_final = out.states[-1]
        # ideal final state
        # identify the 2-qubit component by removing the cavity vacuum: apply U_atom to the atomic part
        # easier: build psi_ideal = tensor(vacuum, U_atom @ atom_vector)
        # build atomic vector:
        atom_state = psi.ptrace([1,2])  # atomic reduced state as density matrix
        # but we want the ket; easier: reconstruct atom ket from psi using tensor product structure
        # Instead, form psi_ideal by acting with U_atom embedded:
        # Build qutip Qobj for U_atom:
        from qutip import Qobj
        Uq = Qobj(U_atom)
        U_full = tensor(qeye(n_cav), Uq)
        psi_ideal = (U_full * psi).unit()
        fid = fidelity(rho_final, psi_ideal.proj())
        fid_list.append(fid)
    return float(np.mean(fid_list)), fid_list

# -------------------------
# Sweep routine
# -------------------------
def sweep_g_and_plot(
    g_scan, omega_c, omega_a, Delta, n_cav,
    kappa, gamma, gamma_phi, plot_coop=True
):
    """
    Sweep g_scan (array of g in rad/s), compute fidelity for iSWAP using dispersive J=g^2/Delta,
    and plot fidelity vs g and fidelity vs C_N (collective cooperativity = N*C_single).
    """
    N_atoms = 2
    psi_basis = atomic_basis_states_as_full(n_cav)

    fidelities = []
    cooperativities = []

    for g in tqdm(g_scan):
        H = build_tc_hamiltonian_two_atoms(g, omega_c, omega_a, n_cav)
        c_ops = build_collapse_ops(n_cav, N=N_atoms, kappa=kappa, gamma=gamma, gamma_phi=gamma_phi)

        # dispersive J estimate (use g as single-atom g)
        J = (g**2) / Delta
        # target iSWAP time
        t_iswap = pi / (2.0 * J)

        # compute fidelity
        avg_fid, fid_list = avg_gate_fidelity_iSWAP(H, c_ops, t_iswap, n_cav)
        fidelities.append(avg_fid)

        # cooperativity (single atom)
        C_single = 4.0 * (g**2) / (kappa * gamma) if (kappa*gamma) > 0 else np.nan
        C_N = N_atoms * C_single
        cooperativities.append(C_N)

        print(f"g/2pi={(g/(2*pi)):.2e} Hz, t_gate={t_iswap:.3e} s, avg_fid={avg_fid:.6f}, C_N={C_N:.2e}")

    # plotting
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].plot(g_scan/(2*pi), fidelities, '-o')
    ax[0].set_xlabel('g / (2π) [Hz]')
    ax[0].set_ylabel('Average iSWAP fidelity')
    ax[0].grid(True)
    if plot_coop:
        ax[1].plot(cooperativities, fidelities, '-o')
        ax[1].set_xscale('log')
        ax[1].set_xlabel('Collective cooperativity C_N')
        ax[1].set_ylabel('Average iSWAP fidelity')
        ax[1].grid(True)
    else:
        ax[1].axis('off')
    plt.tight_layout()
    plt.show()

# -------------------------
# Example parameters & run (USER: replace these with your numbers)
# -------------------------
if __name__ == "__main__":
    # fundamental frequencies (rad/s)
    freq_c_Hz = 10e9   # 10 GHz
    freq_a_Hz = 10e9 + 10e6  # atom detuned by +10 MHz
    omega_c = 2*pi*freq_c_Hz
    omega_a = 2*pi*freq_a_Hz
    Delta = omega_a - omega_c

    # cavity truncation
    n_cav = 8

    # dissipation (rad/s)
    kappa = 2*pi * 1e3       # 1 kHz cavity decay
    gamma = 2*pi * 10.0     # 10 Hz atomic decay
    gamma_phi = 2*pi * 1.0  # 1 Hz dephasing

    # sweep g: from 100 Hz to 200 kHz (in Hz units then converted to rad/s)
    g_hz_scan = np.logspace(2, 5.3, 8)   # 1e2 .. ~2e5 Hz
    g_scan = 2*pi * g_hz_scan  # convert to rad/s

    sweep_g_and_plot(g_scan, omega_c, omega_a, Delta, n_cav, kappa, gamma, gamma_phi)
