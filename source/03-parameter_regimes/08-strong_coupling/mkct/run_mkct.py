# %%
import numpy as np
import matplotlib.pyplot as plt
import yaml

from mkct import MKCT_solver

def do_fft(t, y, zero_padding=0, nblocks=20):
    dt = t[1] - t[0]
    if zero_padding > 0:
        y_blockd = np.array_split(y, nblocks)
        y_blockd_avg = np.array([np.mean(y_block) for y_block in y_blockd])
        y_last = y_blockd_avg[-1]
        y_first = y[0]
        rate_est = (np.abs(y_last) - np.abs(y_first)) / (t[-1] - t[0])

        zeros = y_last * np.exp(np.arange(1, zero_padding + 1)  * rate_est * dt)
        y = np.concatenate((y, zeros))

    w = np.fft.fftfreq(len(y), t[1] - t[0]) * 2 * np.pi
    w = np.fft.fftshift(w)

    Y = np.fft.fft(y)
    Y = np.fft.fftshift(Y) * dt
    return w, Y

def save_time_cmplx(t, C, filename, identifier="") -> None:
    re, im = C.real, C.imag
    datout = np.column_stack((t, re, im))
    fmt = "%15.6e" * 3
    if identifier == "":
        header_list = ['t', 're', 'im']
    else:
        header_list = ['t', f'Re[{identifier}]', f'Im[{identifier}]']

    # header string
    header = f"{header_list[0]:>13}"
    for i in range(1, len(header_list)):
        header += f"{header_list[i]:>15}"

    np.savetxt(filename, datout, fmt=fmt, header=header)






def main():
    tx, re, im = np.loadtxt("../deom/prop-pol-1.dat", unpack=True)
    C_exact = re + 1j * im
    C_exact /= C_exact[0]

    re, im = np.loadtxt("../moments/moments.dat", unpack=True)
    Omega_n = re + 1j * im


    # load the parameters
    with open("../params.yaml", "r") as f:
        params = yaml.safe_load(f)
        rescale = params["scale"]
        Delta = params["Delta"]


    solver = MKCT_solver.init(Omega_n, rescale=rescale)

    # t, C = solver.solve_pade(tf=200, dt=0.001, kernel_order=1, pade_order=(9, 10), conv_domain='time')
    t, C = solver.solve_pade(tf=50, dt=0.0001, kernel_order=1, pade_order=(5,17), conv_domain='frequency')

    K1t = solver.K1t

    fig = plt.figure(dpi=300)
    gs = fig.add_gridspec(2, 1)
    axs = gs.subplots(sharex=True)
    ax = axs[0]
    ax.plot(t, K1t.real, label="K1t (real)")
    ax.axhline(0, color='k', ls='--')
    ax = axs[1]
    ax.plot(t, K1t.imag, label="K1t (imag)")
    ax.axhline(0, color='k', ls='--')
    ax.legend()



    fig = plt.figure(dpi=300)
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

    # Zero padding to imr
    Npad1= C_exact.size * 10
    Npad2 = C.size * 10
    wx, Cwx = do_fft(tx, C_exact, Npad1)
    w, Cw = do_fft(t, C, Npad2)


    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)

    ax.plot((w-Delta), Cw.real, label="MKCT")
    ax.plot((wx-Delta), Cwx.real, color='k', ls='--', label="DEOM")
    ax.axvline(0, color='k', ls='--')
    L = 10
    ax.set_xlim(-L, +L)
    ax.set_xlabel(r"$\omega/\Delta$")
    ax.set_ylabel(r"$I(\omega)$ (arb. units)")
    ax.legend()
    plt.show()

    # Save the results for final production plots
    K1t_original_scale = K1t / rescale**2

    save_time_cmplx(t, K1t_original_scale, "K1t.dat", identifier="K1(t)")
    save_time_cmplx(t, C, "C.dat", identifier="C(t)")




if __name__ == "__main__":
    main()

# %%
