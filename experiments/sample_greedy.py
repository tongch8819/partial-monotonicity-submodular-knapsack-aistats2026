from base_task import BaseTask
from positive_greedy import positive_greedy_lazyupdate2 as positive_greedy
import numpy as np


def sample_greedy_with_prob(model: BaseTask, prob):
    assert prob >= 0. and prob <= 1., "invalid probability"
    greedy_sol = positive_greedy(model, prob=prob)
    sol_fv = greedy_sol['f(S)']
    v_star, v_star_fv = None, float('-inf')
    for e in model.ground_set:
        if model.cost_of_singleton(e) > model.budget:
            continue
        fv = model.objective([e])
        if fv > v_star_fv:
            v_star, v_star_fv = e, fv
    if v_star_fv > sol_fv:
        res = {
            'S': set([v_star]),
            'f(S)': v_star_fv,
            'c(S)': model.cost_of_singleton(v_star),
        }
    else:
        res = {
            'S': greedy_sol['S'],
            'f(S)': sol_fv,
            'c(S)': greedy_sol['c(S)'],
        }
    return res


def sample_greedy(model: BaseTask, delta=1e-2):
    S1 = sample_greedy_with_prob(model, 0.5)
    best_final_S = None
    for S in [S1]:
        if best_final_S is None or S['f(S)'] > best_final_S['f(S)']:
            best_final_S = S
    return best_final_S
