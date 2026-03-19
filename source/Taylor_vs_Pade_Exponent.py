import os
work_path = os.path.dirname(__file__) + '/'
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def taylor_exp(x, n_terms, alpha):
  result = np.zeros_like(x)
  for n in range(n_terms):
    result += np.real((-alpha * x) ** n / math.factorial(n))
  return result
def pade_exp(x, m, n, alpha):
  c = np.array([(-alpha) ** k / math.factorial(k) for k in range(m + n - 1)],
               dtype = complex)
  A = np.array([[c[k - j] for j in range(1, n)] for k in range(m, m + n - 1)])
  b = -c[m:m + n - 1]
  q_tail = np.linalg.solve(A, b)
  q = np.concatenate(([1.0], q_tail))
  p = np.array([sum(q[j] * c[k - j] for j in range(min(k + 1, n)))
                for k in range(m)])
  return np.real(np.polyval(p[::-1], x) / np.polyval(q[::-1], x))

alpha = 2 + 2j
x = np.linspace(-1, 5, 600000)
y = np.exp(-alpha * x)
order_list = [(1,2), (2,3), (3,4), (4,4), (4,5), (5,6), (6,7)]
colors     = ['magenta', 'blue', 'green', 'olive', 'orange', 'red', 'purple']

plt.rc('text', usetex = True)
plt.rc('text.latex', preamble = r'\usepackage{sansmath, xfrac} \sansmath')
with PdfPages(work_path + 'Taylor_vs_Pade_Exponent.pdf') as pdf:
  plt.figure(figsize = (9,6))
  plt.axvline(x = 0, color = 'gray', linestyle = '-.', linewidth = 4)
  for i, (m, n) in enumerate(order_list):
    taylor_y = taylor_exp(x, m + n, alpha)
    plt.plot(x, taylor_y, color = colors[i], linewidth = 2, alpha = .6,
             label = f'$n = {m + n}$')
  plt.plot(x, np.real(y), color = 'black', label = 'Actual Value',
           linewidth = 3, linestyle = '--', alpha = .4)
  plt.xlabel(r'$x$', fontsize = 21)
  plt.ylabel(r'$y = \mathrm e^{-\alpha x}$', fontsize = 21)
  plt.title(r'Taylor Approximations of $y(x) = \exp[-(2 + 2\mathrm i)x]$',
            fontsize = 21)
  plt.legend(fontsize = 15, handlelength = 4)
  plt.grid(True, alpha = .2)
  plt.xlim(-1, 5)
  plt.ylim(-.6, 1.8)
  plt.xticks(fontsize = 18)
  plt.yticks(fontsize = 18)
  pdf.savefig(bbox_inches = 'tight')
  plt.close()
  plt.figure(figsize = (9,6))
  plt.axvline(x = 0, color = 'gray', linestyle = '-.', linewidth = 4)

  for i, (m, n) in enumerate(order_list):
    pade_y = pade_exp(x, m, n, alpha)
    plt.plot(x, pade_y, color = colors[i], linewidth = 2, alpha = .6,
      label =
        f'\\makebox[7.54ex][l]{{$n = {m + n}$:\\,}}$\\sfrac{{[{m}]}}{{[{n}]}}$')
  plt.plot(x, np.real(y), color = 'black', label = 'Actual Value',
           linewidth = 3, linestyle = '--', alpha = .4)
  plt.xlabel(r'$x$', fontsize = 21)
  plt.ylabel(r'$y = \mathrm e^{-\alpha x}$', fontsize = 21)
  plt.title(r'Padé Approximations of $y(x) = \exp[-(2 + 2\mathrm i)x]$',
            fontsize = 21)
  plt.legend(fontsize = 15, handlelength = 4)
  plt.grid(True, alpha = .2)
  plt.xlim(-1, 5)
  plt.ylim(-.6, 1.8)
  plt.xticks(fontsize = 18)
  plt.yticks(fontsize = 18)
  pdf.savefig(bbox_inches = 'tight')
  plt.close()
