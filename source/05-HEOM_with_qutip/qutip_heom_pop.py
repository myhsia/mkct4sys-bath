# %%
import numpy as np
import matplotlib.pyplot as plt
from qutip import (
    about, basis, brmesolve, destroy, expect, liouvillian,
    qeye, sigmax, sigmaz, spost, spre, tensor
)
from qutip.core.environment import OhmicEnvironment
from qutip.solver.heom import HEOMSolver, HSolverDL

def main():
    Delta = 1.0
    Omega = 1.0
    beta = 1.0
    wD = 1.0
    lambd = 0.1
    Lmax = 5

    dt = 0.01
    tf = 50

    sx = sigmax()
    sz = sigmaz()

    Hs = Delta / 2 * sz + Omega / 2 * sx
    V = sz / 2

    psi0 = basis(2, 0)
    rho0 = psi0 * psi0.dag()

    # ohmic spectral density
    env = OhmicEnvironment(
        T=1/beta,
        alpha=2*lambd,
        wc=wD,
        s=1.0,
    )

    # decompose
    tlist = np.linspace(0, 10, 100)
    approx_env, fit_info = env.approximate(
        method="espira-I",
        tlist=tlist,
        Nr=4
    )
    w = np.linspace(0, 5, 1000)
    Jw_approx = approx_env.spectral_density(w)
    Jw_exact = env.spectral_density(w)
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(w, Jw_exact, label="Exact", c='k')
    ax.plot(w, Jw_approx, label="Prony fit", c='r', ls='--')
    ax.set_xlabel(r"$\omega$", fontsize=18)
    ax.set_ylabel(r"J($\omega$)", fontsize=18)
    ax.legend()
    plt.show()

    print(fit_info)

    default_options = {
        "nsteps": 1500,
        "store_states": True,
        "rtol": 1e-12,
        "atol": 1e-12,
        "method": "vern9",
        "progress_bar": "enhanced",
    }

    # use the approximated environment to set up HEOM solver
    HEOMMats = HEOMSolver(Hs, (approx_env,V), max_depth=Lmax, options=default_options)

    # propagate
    tlist = np.linspace(0, tf, int(tf/dt))
    res_heom = HEOMMats.run(rho0, tlist)

    sz_t = expect(sz, res_heom.states)
    sx_t = expect(sx, res_heom.states)
    rho_t = np.zeros((tlist.size, 2, 2), dtype=np.complex128)
    for i, state in enumerate(res_heom.states):
        rho_t[i] = state.ptrace(0).full()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(tlist, sz_t, label=r"$\langle \sigma_z \rangle$")
    ax.plot(tlist, sx_t, label=r"$\langle \sigma_x \rangle$")
    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Expectation values", fontsize=18)
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()



# %%
