# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yaml

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../library"))
from mkct.solver import MKCT_solver
from mkct.runner import do_fft, save_time_cmplx


def main():
    tx, re, im = np.loadtxt("../deom/prop-pol-1.dat", unpack=True)
    C_exact = re + 1j * im
    C_exact /= C_exact[0]

    re, im = np.loadtxt("../moments/moments_quantum.dat", unpack=True)
    Omega_n = re + 1j * im


    # load the parameters
    with open("../params.yaml", "r") as f:
        params = yaml.safe_load(f)
        rescale = params["scale"]
        Delta = params["Delta"]
        pade_m = params["Pade_m"]
        pade_n = params["Pade_n"]


    solver = MKCT_solver.init(Omega_n, rescale=rescale)

    # t, C = solver.solve_pade(tf=200, dt=0.001, kernel_order=1, pade_order=(9, 10), conv_domain='time')
    t, C = solver.solve_pade(tf=100, dt=0.001, kernel_order=1, pade_order=(pade_m, pade_n), conv_domain='frequency')

    K1t = solver.K1t

    with PdfPages('./02-spin_boson_linear_quad.pdf') as pdf:
        # Figure 1. Memory Kernel
        fig = plt.figure()
        gs = fig.add_gridspec(2, 1)
        axs = gs.subplots(sharex=True)
        ax = axs[0]
        ax.plot(t, K1t.real, label="K1t (real)")
        ax.axhline(0, color='k', ls='--')
        ax = axs[1]
        ax.plot(t, K1t.imag, label="K1t (imag)")
        ax.axhline(0, color='k', ls='--')
        ax.legend()
        pdf.savefig(bbox_inches = 'tight')
        plt.close()


        # Figure 2. Autocorrelation Function 
        fig = plt.figure()
        gs = fig.add_gridspec(2, 1)
        axs = gs.subplots(sharex=True)
        ax = axs[0]
        ax.plot(t, C.real, label="MKCT")
        ax.plot(tx, C_exact.real, color='k', label="DEOM", ls='--')
        ax.set_xlim(0, 20)
        ax = axs[1]
        ax.plot(t, C.imag)
        ax.plot(tx, C_exact.imag, color='k', ls='--')
        axs[0].legend()
        pdf.savefig(bbox_inches = 'tight')
        plt.close()


        # Figure 3. Absorption Lineshape Function
        # Zero padding to imr
        Npad1= C_exact.size * 10
        Npad2 = C.size * 10
        wx, Cwx = do_fft(tx, C_exact, Npad1)
        w, Cw = do_fft(t, C, Npad2)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot((-w-Delta), Cw.real, label="MKCT")
        ax.plot((-wx-Delta), Cwx.real, color='k', ls='--', label="DEOM")
        ax.axvline(0, color='k', ls='--')
        w0 = 0.3
        L = 2.5
        ax.set_xlim(w0-L, w0+L*1.2)
        ax.set_xlabel(r"$\omega/\Delta$")
        ax.set_ylabel(r"$I(\omega)$ (arb. units)")
        ax.legend()
        pdf.savefig(bbox_inches = 'tight')
        plt.close()

    # Save the results for final production plots
    K1t_original_scale = K1t / rescale**2

    save_time_cmplx(t, K1t_original_scale, "K1t.dat", identifier="K1(t)")
    save_time_cmplx(t, C, "C.dat", identifier="C(t)")




if __name__ == "__main__":
    main()

# %%
