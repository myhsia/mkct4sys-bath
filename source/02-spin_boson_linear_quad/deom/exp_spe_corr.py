# %%
import numpy as np
import matplotlib.pyplot as plt
# import scienceplots
import sympy as sp
import yaml
from scipy.integrate import quad

import os
work_path = os.path.dirname(__file__) + '/'

# plt.style.use(['science', 'grid'])

from prony.prony import prony
from prony.TimeDomainData import TimeDomainData

from prony.fitting import (
    get_gammas_and_t,
    get_gamma_matrix,
    get_correlation_function_matrix,
    get_freq_matrix,
    get_expn,
    optimize
)
from prony.spectral import get_spectral_function_from_exponentials
from prony.spectral import bose_function
from prony.deom import get_symmetrized_deom_inputs
from spin_lattice_utils.third_party.deom import complex_2_json, convert, init_qmd, init_qmd_quad
from spin_lattice_utils.third_party.deom import decompose_spe
from spin_lattice_utils.third_party.deom import decompose_spe_prony_na

import json

# Load the parameters from the YAML file
with open(work_path + "../params.yaml", "r") as f:
    params = yaml.safe_load(f)

rescale = params["scale"]

# Define the parameters
Nmax_Omega = params["Nmax_Omega"]
Nmax_tilde_Omega = params["Nmax_tilde_Omega"]

beta = params["beta"]
Omega = params["Omega"]
Delta = params["Delta"]
lambd = params["lambd"]
thetaB = params["theta_B"]
wB = params["wB"]
wD = params["wD"]
eta = params["eta"]

alp0 = lambd * thetaB**2
alp1 = -np.sqrt(2 * lambd * wB) * thetaB**2
alp2 = wB * (thetaB**2 - 1) / 2

print(f"{alp2=}, {alp1=}, {alp0=}")

raise SystemExit


sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])

# Hs = Delta / 2 * sz + Omega / 2 * sx
Hs = np.array([[0, 0], [0, Delta]])
V = np.array([[0, 0], [0, 1]])
# mu = sx / 2
mu = sx
rho0 = np.diagflat([1.0, 0.0])

# parameters (HEOM code)
nmax = 1000000
nmod = 1
temp = 1 / beta
OMG = 50
lmax = 25
ferr = 1.0E-12
tf = 100
dt = 0.002
npsd = 3


def J(w):
    return 2 * eta * w * np.exp(-np.abs(w) / wD)

def main():
    w = np.linspace(0, 10, 1000)
    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot(111)
    # ax.plot(w, J(w))
    # ax.set_xlabel(r"$\omega$")
    # ax.set_ylabel(r"$J(\omega)$")
    # plt.show()

    # zihao's prony

    w_sp = sp.symbols('w', real=True)
    wD_sp, beta_sp, eta_sp = sp.symbols('wD beta eta', positive=True)
    spe = 2 * eta_sp * w_sp * sp.exp(-sp.Abs(w_sp) / wD_sp)
    sp_para_dict = {
        wD_sp: wD,
        beta_sp: beta,
        eta_sp: eta
    }
    para_dict = {
        'beta': beta,
    }
    condition_dict = {}
    nind = [1, npsd]
    etal, etar, etaa, expn = decompose_spe_prony_na(
        spe, w_sp, sp_para_dict, para_dict, condition_dict, nind,
        scale=30.0, n=1000, scale_fft=2000)

    len_ = 10000
    spe_wid = 200
    w = np.append(np.linspace(-spe_wid, 0, len_), np.linspace(0, spe_wid, len_))
    jw_exact = J(w) * bose_function(w, beta)
    jw_prony = get_spectral_function_from_exponentials(w, expn, etal)

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(w, jw_exact, label='Exact')
    ax.plot(w, jw_prony, label='Prony')
    ax.legend()
    ax.set_xlim(-5, 15)
    # plt.show()

    # data = TimeDomainData(J, bose_function, beta, is_fermi=False, tf=10, n_sample=10000000, n_Hankel=100, max_freq=3000*wD)
    # data.plot_correlation_function(data.time, data.correlation_function)

    # nmode_real = 1
    # nmode_imag = npsd
    # expn, etal = prony(data, nmode_real, nmode_imag)

    # len_ = 10000
    # spe_wid = 200
    # w = np.append(np.linspace(-spe_wid, 0, len_), np.linspace(0, spe_wid, len_))
    # jw_exact = J(w) * bose_function(w, beta)
    # jw_prony = get_spectral_function_from_exponentials(w, expn, etal)

    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot(111)
    # ax.plot(w, jw_exact, label='Exact')
    # ax.plot(w, jw_prony, label='Prony')
    # ax.legend()
    # ax.set_xlim(-10, 30)
    # plt.show()

    # etal, etar, etaa, expn = get_symmetrized_deom_inputs(etal, expn)

    # # zihao's prony
    # w_sp = sp.symbols('w', real=True)
    # wD_sp, beta_sp, lambd_sp = sp.symbols('wD beta lambda', positive=True)
    # spe = 2 * lambd_sp * w_sp * sp.exp(-sp.Abs(w_sp) / wD_sp)
    # sp_para_dict = {
    #     wD_sp: wD,
    #     beta_sp: beta,
    #     lambd_sp: lambd
    # }
    # para_dict = {
    #     'beta': beta,
    # }
    # condition_dict = {}
    # nind = [1, npsd]
    # etal, etar, etaa, expn = decompose_spe_prony_na(
    #     spe, w_sp, sp_para_dict, para_dict, condition_dict, nind, scale=2.0)

    # len_ = 10000
    # spe_wid = 200
    # w = np.append(np.linspace(-spe_wid, 0, len_), np.linspace(0, spe_wid, len_))
    # jw_exact = J(w) * bose_function(w, beta)
    # jw_prony = get_spectral_function_from_exponentials(w, expn, etal)

    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot(111)
    # ax.plot(w, jw_exact, label='Exact')
    # ax.plot(w, jw_prony, label='Prony')
    # ax.legend()
    # ax.set_xlim(-3, 3)
    # plt.show()




    mode = np.zeros_like(expn, dtype=int)
    qmds = np.array([V])

    twsg = np.zeros((len(expn), nmod))
    for i in range(len(expn)):
        for j in range(nmod):
            twsg[i, j] = i + j * len(expn)

    sdip = np.zeros((nmod, 2, 2), dtype=float)
    pdip = np.zeros((nmod, 2, 2), dtype=float)
    bdip = np.zeros(nmod * len(expn), dtype=float)
    sdip[0, :, :] = mu

    pdip2_temp = np.zeros((nmod, 2, 2), dtype=float)
    pdip2_temp[0, :, :] = mu
    pdip2 = np.zeros((nmod, nmod, 2, 2), dtype=complex)
    for i in range(nmod):
        for j in range(nmod):
            pdip2[i, j, :, :] = pdip2_temp[i, :, :]


    # For quadratic coupling
    qmd2 = np.zeros((nmod, nmod, 2, 2), dtype=complex)
    for i in range(nmod):
        for j in range(nmod):
            qmd2[i, j, :, :] = qmds[i, :, :]

    json_init = {
        "syl": {
            "OMG": OMG,
            "nind": len(expn),
            "lwsg": nmod,
            "twsg": list(twsg.flatten())
        },
        "nmax": nmax,
        "lmax": lmax,
        "alp0": alp0,
        "alp1": alp1,
        "alp2": alp2,
        "ferr": ferr,
        "filter": True,
        "nind": len(expn),
        "nmod": nmod,
        "equilibrium": {
            "sc2": True,
            "dt-method": False,
            "OMG": OMG,
            "ti": 0,
            # "tf": 25,
            "tf": 0,
            "dt": dt,
        },
        "renormalize": complex_2_json((alp0 + alp2 * np.sum(etal)) * qmds),
        # "renormalize": complex_2_json(alp0 * qmds),
        # "renormalize": complex_2_json((alp0 + alp2 * get_renormalization_energy()) * qmds),
        "expn": complex_2_json(expn),
        "ham1": complex_2_json(Hs),
        "coef_abs": complex_2_json(etaa),
        "spectrum": True,
        "spectrum-data": {
            "dipole": {
                "sdip_cub": complex_2_json(sdip),
                "bdip1_cub": complex_2_json(bdip),
            },
            "dipole1": {
                "sdip_cub": complex_2_json(sdip),
                "bdip1_cub": complex_2_json(bdip),
            },
            "if-time": True,
            "time": {
                "lcr": 'l',
                "Hei": False,
                "noise": True,
                "ti": 0,
                "tf": tf,
                "dt": dt,
                "filter_ferr": ferr,
            },
            "file": "prop-pol-1.dat",
        },
    }

    init_qmd(json_init, qmds, qmds, mode, 2, etaa, etal, etar)
    init_qmd_quad(json_init, qmd2, qmd2, qmd2, mode, 2, len(expn), nmod, etaa, etal, etar)

    init_qmd(json_init["spectrum-data"]["dipole"], pdip, pdip, mode, 2, etaa, etal, etar)
    init_qmd_quad(json_init["spectrum-data"]["dipole"], pdip2, pdip2, pdip2, mode, 2, len(expn), nmod, etaa, etal, etar)

    init_qmd(json_init["spectrum-data"]["dipole1"], pdip, pdip, mode, 2, etaa, etal, etar)
    init_qmd_quad(json_init["spectrum-data"]["dipole1"], pdip2, pdip2, pdip2, mode, 2, len(expn), nmod, etaa, etal, etar)

    with open(work_path + "input.json", 'w') as f:
        json.dump(json_init, f, indent=2, default=convert)



if __name__ == "__main__":
    main()



# %%
