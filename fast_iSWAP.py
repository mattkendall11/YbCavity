"""
fast_iSWAP_fixed.py

Faster sweep of single-atom coupling g using Liouvillian + expm_multiply
(single-time propagation). This version fixes the QuTiP -> SciPy sparse
conversion issue and is robust to fallback cases.

Requirements:
 - numpy, matplotlib, tqdm, scipy, qutip

Notes:
 - All frequencies and rates are in angular units (rad/s).
 - Convert Hz to rad/s with omega = 2*pi*freq_Hz.
 - Uses interaction-frame Hamiltonian H_cav = Delta * a^dag a.
"""

import numpy as np
from math import pi
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# QuTiP imports
from qutip import (
    basis, tensor, qeye, destroy, sigmap, sigmam, sigmaz,
    liouvillian, operator_to_vector, vector_to_operator, Qobj
)

# SciPy efficient exponential action & sparse
from scipy.sparse.linalg import expm_multiply
import scipy.sparse as sp

# -----------------------
# User-tunable parameters
# -----------------------
# Physical frequencies (Hz) and conversion to rad/s done below.
freq_c_Hz = 10e9                # cavity freq in Hz (10 GHz)
detuning_Hz = 10e6              # atom is +10 MHz detuned relative to cavity (Hz)
freq_a_Hz = freq_c_Hz + detuning_Hz

# Convert to angular frequency (rad/s)
omega_c = 2 * pi * freq_c_Hz
omega_a = 2 * pi * freq_a_Hz
Delta = omega_a - omega_c  # should equal 2*pi*detuning_Hz

# Cavity truncation
n_cav = 8   # increase to check convergence (monitor <a^dag a>)

# Dissipation rates (Hz -> rad/s)
kappa = 2 * pi * 1e3      # cavity decay 1 kHz
gamma = 2 * pi * 10.0     # atomic population decay 10 Hz
gamma_phi = 2 * pi * 1.0  # pure dephasing 1 Hz

# Sweep parameters (g in Hz, will be converted to rad/s)
g_hz_scan = np.logspace(2, 5.3, 12)   # 1e2 .. ~2e5 Hz (12 points)
g_scan = 2 * pi * g_hz_scan          # rad/s

# Safety threshold: skip g values producing gate time longer than this (seconds)
t_max = 5e-3   # 5 ms; adjust as you like

# Useful constants
N_atoms = 2

# -----------------------
# Helper: robust propagate via expm_multiply
# -----------------------
def propagate_via_expm(L_qobj: Qobj, rho0_qobj: Qobj, t: float):
    """
    Propagate density matrix rho0 under Liouvillian L_qobj for time t using
    scipy.sparse.linalg.expm_multiply. Returns rho_final (Qobj).
    Converts QuTiP internal sparse representation to scipy.sparse.csr_matrix
    if necessary; falls back to dense conversion if needed.
    """
    # Convert QuTiP Liouvillian.data to a scipy.sparse matrix
    try:
        # QuTiP CSR wrapper often supports .tocsr()
        Lmat = L_qobj.data.tocsr()
    except Exception:
        # Fall back to dense conversion (memory-heavy), but handle gracefully
        warnings.warn("Falling back to dense conversion of Liouvillian matrix. "
                      "This may be memory intensive.")
        Lmat = sp.csr_matrix(L_qobj.data.toarray())

    # Convert rho0 to a 1D numpy vector expected by expm_multiply
    vec0_qobj = operator_to_vector(rho0_qobj)   # returns a Qobj column vector
    vec0 = np.asarray(vec0_qobj.full()).ravel()

    # Compute exp(L * t) @ vec0 using Krylov methods
    vec_final = expm_multiply(Lmat * t, vec0)

    # Convert back to Qobj density matrix
    rho_final = vector_to_operator(vec_final)
    return rho_final

# -----------------------
# Precompute objects (once)
# -----------------------
# cavity operators
a = destroy(n_cav)
adag = a.dag()

# two-level atom local operators
sm = sigmam()
sp = sigmap()
sz = sigmaz()
id2 = qeye(2)

# atomic two-qubit operators in atoms-only space
sm_0_atoms = tensor(sm, id2)
sm_1_atoms = tensor(id2, sm)
sp_0_atoms = tensor(sp, id2)
sp_1_atoms = tensor(id2, sp)
sz_0_atoms = tensor(sz, id2)
sz_1_atoms = tensor(sz, id2)
id_atoms = tensor(id2, id2)

# promote cavity operators to full Hilbert space (cavity x atoms)
a_full = tensor(a, id_atoms)
adag_full = a_full.dag()

# H_cav in the interaction frame (H_cav = Delta * a^dag a)
H_cav = Delta * adag_full * a_full

# precompute interaction template (multiplied by g)
sm0_full = tensor(qeye(n_cav), sm_0_atoms)
sp0_full = tensor(qeye(n_cav), sp_0_atoms)
sm1_full = tensor(qeye(n_cav), sm_1_atoms)
sp1_full = tensor(qeye(n_cav), sp_1_atoms)
H_int_template = (adag_full * sm0_full + a_full * sp0_full) + (adag_full * sm1_full + a_full * sp1_full)

# Precompute collapse operator templates (do not depend on g)
c_ops_template = []
if kappa > 0:
    c_ops_template.append(np.sqrt(kappa) * a_full)
if gamma > 0:
    c_ops_template.append(np.sqrt(gamma) * tensor(qeye(n_cav), sm_0_atoms))
    c_ops_template.append(np.sqrt(gamma) * tensor(qeye(n_cav), sm_1_atoms))
if gamma_phi > 0:
    # NOTE: check dephasing prefactor convention for your T_phi mapping
    c_ops_template.append(np.sqrt(gamma_phi) * tensor(qeye(n_cav), sz_0_atoms))
    c_ops_template.append(np.sqrt(gamma_phi) * tensor(qeye(n_cav), sz_1_atoms))

# computational basis (full space with cavity vacuum)
vac = basis(n_cav, 0)
g_ket = basis(2, 0)
e_ket = basis(2, 1)
psi00 = tensor(vac, tensor(g_ket, g_ket))
psi01 = tensor(vac, tensor(g_ket, e_ket))
psi10 = tensor(vac, tensor(e_ket, g_ket))
psi11 = tensor(vac, tensor(e_ket, e_ket))
psi_basis = [psi00, psi01, psi10, psi11]

# ideal iSWAP on atomic subspace embedded in full Hilbert space
U = np.zeros((4, 4), dtype=complex)
U[0, 0] = 1.0
U[3, 3] = 1.0
U[1, 2] = 1j
U[2, 1] = 1j
Uq = Qobj(U)
U_full = tensor(qeye(n_cav), Uq)

# -----------------------
# Sweep loop (Liouvillian + expm_multiply)
# -----------------------
fidelities = []
cooperativities = []

print("Starting sweep over g values (outer progress bar)...")
for g in tqdm(g_scan, desc="g-scan"):
    # Build H quickly from precomputed pieces
    H = H_cav + g * H_int_template

    # Gate time using dispersive J = g^2 / Delta --> iSWAP t = pi / (2J)
    # Note: Valid only in dispersive limit |Delta| >> g, kappa
    J = (g ** 2) / Delta
    if J == 0:
        fidelities.append(np.nan)
        cooperativities.append(np.nan)
        continue
    t_iswap = pi / (2.0 * J)

    # skip enormous gate times
    if t_iswap > t_max:
        fidelities.append(np.nan)
        C_single = 4.0 * (g ** 2) / (kappa * gamma) if (kappa * gamma) > 0 else np.nan
        cooperativities.append(N_atoms * C_single)
        continue

    # Build Liouvillian (Qobj superoperator)
    L = liouvillian(H, c_ops_template)

    # For each computational-basis input, propagate with expm_multiply(L * t) * vec(rho0)
    fid_list = []
    for psi in psi_basis:
        rho0 = psi.proj()
        try:
            rho_final = propagate_via_expm(L, rho0, t_iswap)
        except Exception as ex:
            # if expm_multiply route fails for unexpected reason, fall back to mesolve
            warnings.warn(f"expm_multiply propagation failed with error: {ex}. Falling back to mesolve for this point.")
            from qutip import mesolve
            out = mesolve(H, rho0, [0.0, t_iswap], c_ops=c_ops_template, options=None)
            rho_final = out.states[-1]

        # ideal final state
        psi_ideal = (U_full * psi).unit()

        # fidelity of rho_final with the pure ideal state |psi_ideal><psi_ideal|
        fid = (rho_final * psi_ideal.proj()).tr().real
        fid_list.append(float(np.real_if_close(fid)))

    avg_fid = float(np.mean(fid_list))
    fidelities.append(avg_fid)

    # cooperativity (single atom) and collective
    C_single = 4.0 * (g ** 2) / (kappa * gamma) if (kappa * gamma) > 0 else np.nan
    C_N = N_atoms * C_single
    cooperativities.append(C_N)

    print(f"g/2π = {g/(2*pi):.3e} Hz, t_gate = {t_iswap:.3e} s, avg_fid = {avg_fid:.6f}, C_N = {C_N:.3e}")

# -----------------------
# Plot results
# -----------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(g_hz_scan, fidelities, '-o')
ax[0].set_xscale('log')
ax[0].set_xlabel('g / (2π) [Hz]')
ax[0].set_ylabel('Average iSWAP fidelity')
ax[0].grid(True)

ax[1].plot(cooperativities, fidelities, '-o')
ax[1].set_xscale('log')
ax[1].set_xlabel('Collective cooperativity C_N')
ax[1].set_ylabel('Average iSWAP fidelity')
ax[1].grid(True)

plt.tight_layout()
plt.show()
