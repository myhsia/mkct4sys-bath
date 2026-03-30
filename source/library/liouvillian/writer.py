import numpy as np

class MomentsWriter:
    def __init__(self, filename, flag='moments_quantum'):
        self.filename = filename
        self.f = open(filename, 'w')

        # write the header
        if flag == 'tilde_moments_quantum':
            RE_str = "Re[tilde_Omega_n]"
            IM_str = "Im[tilde_Omega_n]"
        elif flag == 'moments_quantum':
            RE_str = "Re[Omega_n]"
            IM_str = "Im[Omega_n]"
        else:
            raise ValueError(f"Invalid flag {flag}. Available flags are 'K1n' and 'moments_quantum'.")

        self.HEADER = f"# {RE_str:>18} {IM_str:>20}\n"

        # write the header
        self.f.write(self.HEADER)

    def write_line(self, val):
        re, im = np.real(val), np.imag(val)
        re_str = f"{re:20.10E}"
        im_str = f"{im:20.10E}"
        self.f.write(f"{re_str} {im_str}\n")
        # Manually flush the buffer
        self.f.flush()

    def close(self):
        self.f.close()