from base_task import BaseTask
from positive_greedy import positive_greedy_lazyupdate2 as positive_greedy


def positive_modified_greedy(model: BaseTask):
    greedy_sol = positive_greedy(model)
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
