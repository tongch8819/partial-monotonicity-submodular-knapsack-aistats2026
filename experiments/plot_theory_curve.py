from typing import Callable, Iterable
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times New Roman'


def gen_m_y_vals(func: Callable, iterative: bool = False, debug=False, send_in_args={}):
    """
    Wrapper function to generate approximation ratios
    """
    prec = 0.01
    ms = np.arange(0., 1.+prec, prec)
    epsilon = 1e-6
    ms[0] += epsilon
    debug_dict = {}
    if iterative:
        res = np.zeros_like(ms)
        for i, m in enumerate(ms):
            res[i] = func(m)
    else:
        if not debug:
            res = func(ms, **send_in_args)
        else:
            res, debug_dict = func(ms, debug=True, **send_in_args)
    if not debug:
        return ms, res
    else:
        return ms, res, debug_dict


def compute_m_prime(m: Iterable, llambda):
    res = m + (m * m - m) / (llambda - m)
    return res


def kuller_func(m: Iterable):
    res = m * (1 - 1 / np.e)
    return res / 2


def revisit_small_func(m: Iterable, llambda, debug=False):
    left = m * (llambda - 1)
    left /= llambda + m * (llambda - 1) - m
    right = m * m * (1 - 1 / np.sqrt(np.e))
    return np.minimum(left, right)


def revisit_small_func_closure(m: Iterable):
    res = np.zeros_like(m)
    for l in list(np.linspace(1.001, 1.3, 20)) + list(np.linspace(1.4, 3.0, 5)):
        partial_solution = revisit_small_func(m, l, debug=False)
        res = np.maximum(partial_solution, res)
    return res


def greedy_max_func(m: Iterable):
    return m / 2


def two_set_enum_tight_func(m, llambda, debug=False):
    alpha = 1 - np.exp(-1)
    m_prime = m + (m * m - m) / (llambda - m)
    res = alpha * m_prime
    h_m = m / llambda - 3 * alpha / 2 + alpha * m * (1 - m_prime) / llambda
    h_m = np.minimum(h_m, 0)
    res += h_m
    if debug:
        debug_dict = {"h(m)": h_m, }
        return res, debug_dict
    else:
        return res


def two_set_enum_tight_func_new(m, llambda, debug=False):
    alpha = 1 - np.exp(-1)
    m_prime = m + (m * m - m) / (llambda - m)
    res = alpha * m_prime
    h_m = m * (2 - m_prime) / llambda
    h_m += (3 / 2) * (- alpha * m_prime + m_prime - 1)
    h_m = np.minimum(h_m, 0)
    res += h_m
    if debug:
        debug_dict = {"h(m)": h_m, }
        return res, debug_dict
    else:
        return res


def two_set_enum_tight_closure(m, help_func=two_set_enum_tight_func, debug=False):
    res = np.zeros_like(m)
    for l in list(np.linspace(1.001, 1.19, 5)) + list(np.linspace(1.2, 3.0, 5)):
        Y_partial_solution = help_func(m, l, debug=False)
        res = np.maximum(Y_partial_solution, res)
    return res


def one_set_enum_func(m: Iterable, llambda=1.1):
    m_prime = compute_m_prime(m, llambda)
    left = m_prime / 2
    left += m * (2 - m_prime) / (8 * llambda)
    right = m_prime * (1 - np.exp(-1))
    res = np.minimum(left, right)
    return res


def sample_greedy_func(m: Iterable):
    threshold = 1 / 5
    adjustment = 0.0001
    left = (m + 1) / 6
    common = np.sqrt((m - 2) * (m - 1))
    right = - (m + common - 2) * (m + common - 1)
    right /= common + 1e-5
    left[m <= threshold - adjustment] = 0
    right[m >= threshold + adjustment] = 0
    res = np.maximum(left, right)
    correct_af_nonmonotone = 1 / (3 + 2 * np.sqrt(2))
    assert abs(res[0] - correct_af_nonmonotone) <= 1e-4
    return res


def inapproximability_func(m_vals: Iterable, prec_num=100):
    vals = []
    for m in m_vals:
        y = 1
        for alpha in np.linspace(0, 1, prec_num):
            x = np.linspace(0, 1, prec_num)
            numerator = alpha * (m * x * x + 2 * x - 2 * x * x) + 2 * \
                (1 - alpha) * (1 - np.exp(x - 1)) * (1 - (1 - m) * x)
            numerator = np.max(numerator)
            denom = np.maximum(1, 2 * (1 - alpha))
            y = np.minimum(y, numerator / denom)
        vals.append(y)
    return vals


X, Y_revisit_small = gen_m_y_vals(revisit_small_func_closure)
X, Y_khuller = gen_m_y_vals(kuller_func)
X, Y_greedy_max = gen_m_y_vals(greedy_max_func)
X, Y_two_set_enum = gen_m_y_vals(two_set_enum_tight_closure, send_in_args={
                                 "help_func": two_set_enum_tight_func})
X, Y_one_set_enum = gen_m_y_vals(one_set_enum_func)
X, Y_sample_greedy = gen_m_y_vals(sample_greedy_func)
X, Y_inapproximability = gen_m_y_vals(inapproximability_func)
show = False
algo_names_lst = [
    "Positive Modified Greedy",
    "Positive Greedy+Max",
    "Two Set Enumeration Positive Greedy",
    "One Set Enumeration Positive Greedy+Max",
    "Sample Greedy",
    "Inapproximability",
]
posi_mgreedy_af = np.maximum(Y_khuller, Y_revisit_small)
algo_to_af_dict = {
    "Positive Modified Greedy": posi_mgreedy_af,
    "Positive Greedy+Max": Y_greedy_max,
    "Two Set Enumeration Positive Greedy": np.maximum(posi_mgreedy_af, Y_two_set_enum),
    "One Set Enumeration Positive Greedy+Max": np.maximum(Y_one_set_enum, Y_greedy_max),
    "Sample Greedy": np.maximum(Y_sample_greedy, posi_mgreedy_af),
}
prec = 0.01
ms = np.arange(0., 1.+prec, prec)


def sub_sampling(x, y, factor=4):
    new_x = x[::factor]
    new_y = y[::factor]
    return new_x, new_y


t_markers = ['o', 's', 'D', '^', 'v', '<', '+']
markers = {n: m for m, n in zip(t_markers, algo_names_lst)}
fig = plt.figure(figsize=(6, 5))
for i, (algo_name, af) in enumerate(algo_to_af_dict.items()):
    new_ms, new_af = sub_sampling(ms, af)
    plt.plot(new_ms, new_af, label=algo_name,
             marker=markers[algo_name], alpha=0.5)
plt.xlabel("Monotonicity Ratio", fontsize=20)
plt.ylabel("Approximation Ratio", fontsize=20)
plt.legend(loc="upper left", fontsize=12)
plt.ylim(0, 0.7)
plt.tight_layout()
if show:
    plt.show()
else:
    plt.savefig("result/afs_algs.pdf", format="pdf", bbox_inches='tight')
