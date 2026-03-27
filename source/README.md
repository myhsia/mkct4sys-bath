# Supporting information for "Universal Structure of Computing Moments for Exact Quantum Dynamics: Application to Arbitrary System-Bath Couplings"

Source code implementing the methods for computing higher-order moments as described in the paper "Universal Structure of Computing Moments for Exact Quantum Dynamics: Application to Arbitrary System-Bath Couplings". This code provides a general framework for evaluating time derivatives of quantum correlation functions through exact moment expansions, enabling applications across a wide range of open quantum system models and coupling regimes.

## Directory structure and files
The directory structure is as follows:

```text
.
├── 00-spin_boson_linear
├── 01-spin_boson_quad
├── 02-spin_boson_linear_quad
├── 03-parameter_regimes
├── 04-example_mkct_thirdparty
├── README.md
└── src
```

### Source code 
The source code for computing the moments is located in the `src`. The directory structure and simple description of the files are as follows:

```text
.
├── apply_iLv.py         # wrappers for applying the Liouvillian operator and evaluating the Mori inner products
├── bath_mode.py         # class for the bath modes Qm (Eq. 21) and Pn (Eq. 22).
├── bath_polynomial.py   # class for the bath polynomial terms in the general term (Eq. 20)
├── expval_bath_poly.py  # symbolic evaluation of the bath polynomial expected value using the generating function (Eq. 32)
├── poly_moments.py      # The main function for computing the moments 
├── sb_general_term.py   # class for the general term (Eq. 20)
├── utils.py             # Miscellaneous utility functions
└── writer.py            # class for writing the output files
```


### Workspaces for the figures in the paper

Here, the raw data files to create the figures in the paper are located:
- `00-spin_boson_linear`: Spin-boson model with linear coupling. Corresponds to FIGs. 2 and 3 in the paper.
- `01-spin_boson_quad`: Spin-boson model with quadratic coupling. Corresponds to FIGs. 4 and 6 in the paper.
- `02-spin_boson_linear_quad`: Spin-boson model with both linear and quadratic couplings. Corresponds to FIG. 5 and 6 in the paper.
- `03-parameter_regimes`: Spin-boson model with linear coupling in different parameter regimes. Corresponds to FIGs. 7, 8, and 9 in the paper. 
 
In each of these workspaces, the directory structure is as follows:

``` text
.
├── deom        # DEOM input and output files
├── mkct        # MKCT input and output files
├── moments     # Example of using `src/poly_moments.py` to compute the moments
└── params.yaml # Parameters for the spin-boson model
```

### Third-party libraries
The paper uses MKCT to post-process the moments computed by the code. A helper repository is written by the authors and is avialable at `thirdparty/mkct`. 

The DEOM code we used is developed by our collaborators and is available at GitLab repository [`moscal2.0`](https://git.lug.ustc.edu.cn/czh123/moscal2.0.git).



