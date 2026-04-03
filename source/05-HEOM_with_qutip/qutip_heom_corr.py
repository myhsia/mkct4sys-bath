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
    Omega = 0.0
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
    # to calculate <A(t) B(0)>,
    # we evolve B*rho0 forward in time and measure A at each time step
    A = B = mu = sx

    Brho = B * rho0
    tlist = np.arange(0, tf+dt, dt)
    heom_res = HEOMMats.run(Brho, tlist)
    C_AB = expect(A, heom_res.states)



    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(tlist, C_AB.real, label=r"Re$\langle A(t)B(0)\rangle$", c='b')
    ax.plot(tlist, C_AB.imag, label=r"Im$\langle A(t)B(0)\rangle$", c='r')
    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Expectation values", fontsize=18)
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()



# %%
