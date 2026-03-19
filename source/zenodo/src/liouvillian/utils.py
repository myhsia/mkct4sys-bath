import numpy as np

def parse_reps(m_list):
    m_set = sorted(list(set(m_list)))
    m_reps = [0] * len(m_set)
    for ii, m in enumerate(m_set):
        for mm in m_list:
            if m == mm:
                m_reps[ii] += 1
    return m_set, m_reps

def comm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.dot(A, B) - np.dot(B, A) 

def tab_str(s, N=69, char='=', border='!'):
    n = len(s)
    n_left = (N - n - 4) // 2
    n_right = N - n - n_left
    return border + char * n_left + ' ' + s + ' ' + char * n_right + border

