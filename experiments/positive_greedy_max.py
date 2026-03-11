from base_task import BaseTask
from typing import List, Set
from copy import deepcopy
import numpy as np


def positive_greedy_max_lazyupdate(model: BaseTask):
    """
    positive greedy max algorithm
    """
    augmented_soluton_set = []
    solution_set = []
    cur_cost = 0.
    N = list(model.ground_set)
    pq = sorted([(e, model.density(e, solution_set))
                for e in N], key=lambda x: -x[1])
    solution_set.append(pq[0][0])
    cur_cost += model.cost_of_singleton(pq[0][0])
    pq = pq[1:]
    lazy_idx = 1
    not_valid = set()
    while len(not_valid) < len(N):
        valid_elements = list(set(N) - not_valid)
        pq_marginal_gain = sorted([(e, model.marginal_gain(
            e, solution_set)) for e in valid_elements], key=lambda x: -x[1])
        augmented_soluton_set = deepcopy(
            solution_set) + [pq_marginal_gain[0][0]]
        if lazy_idx >= len(pq):
            lazy_idx = len(pq) - 1
        current_potential_best_element, current_potential_best_density = pq[lazy_idx]
        current_potential_best_density = model.density(
            current_potential_best_element, solution_set)
        if lazy_idx + 1 >= len(pq):
            previous_next_best_density = -1
        else:
            _, previous_next_best_density = pq[lazy_idx + 1]
        if current_potential_best_density >= previous_next_best_density:
            lazy_idx += 1
            current_best_element = current_potential_best_element
            current_best_density = current_potential_best_density
        else:
            pq = list(filter(lambda x: x[0] in set(solution_set), pq))
            pq = sorted([(e, model.density(e, solution_set))
                        for e in N], key=lambda x: -x[1])
            lazy_idx = 1
            current_best_element, current_best_density = pq[0]
        if current_best_element in not_valid:
            continue
        max_density = current_best_density
        if max_density < 0.:
            break
        solution_set.append(current_best_element)
        cur_cost += model.cost_of_singleton(current_best_element)
        for e, d in pq:
            if e == current_best_element or model.cost_of_singleton(e) + cur_cost > model.budget:
                not_valid.add(e)
    S, G = augmented_soluton_set, solution_set
    S_fv = model.objective(S)
    G_fv = model.objective(G)
    if S_fv >= G_fv:
        res = {
            'S': S,
            'f(S)': S_fv,
            'c(S)': model.cost_of_set(S),
        }
    else:
        res = {
            'S': G,
            'f(S)': G_fv,
            'c(S)': model.cost_of_set(G),
        }
    return res


def positive_greedy_max_original(model: BaseTask):
    """
    positive greedy max algorithm
    """
    G, S = set(), set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.
    while len(remaining_elements):
        s, max_marginal_gain = None, -1
        for e in remaining_elements:
            mg = model.marginal_gain(e, G)
            if s is None or mg > max_marginal_gain:
                s, max_marginal_gain = e, mg
        assert s is not None
        tmp_G = deepcopy(G)
        tmp_G.add(s)
        if model.objective(S) < model.objective(tmp_G) and model.cost_of_set(tmp_G) <= model.budget:
            S = tmp_G
        a, max_density = None, -1.
        for e in remaining_elements:
            ds = model.density(e, G)
            if a is None or ds > max_density:
                a, max_density = e, ds
        assert a is not None
        if cur_cost + model.cost_of_singleton(a) <= model.budget:
            G.add(a)
            cur_cost += model.cost_of_singleton(a)
        remaining_elements.remove(a)
        to_remove = set()
        for v in remaining_elements:
            if model.cost_of_singleton(v) + cur_cost > model.budget:
                to_remove.add(v)
        remaining_elements -= to_remove
    S_fv = model.objective(S)
    G_fv = model.objective(G)
    if S_fv >= G_fv:
        res = {
            'S': S,
            'f(S)': S_fv,
            'c(S)': model.cost_of_set(S),
        }
    else:
        res = {
            'S': G,
            'f(S)': G_fv,
            'c(S)': model.cost_of_set(G),
        }
    return res
