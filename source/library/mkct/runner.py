import numpy as np


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
