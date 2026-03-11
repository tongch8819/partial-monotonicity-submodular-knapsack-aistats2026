from base_task import BaseTask
import numpy as np


def positive_greedy_lazyupdate2(model: BaseTask, prob: float = None):
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
        if prob is not None and np.random.uniform(0., 1.) >= prob:
            continue
        else:
            solution_set.append(current_best_element)
            cur_cost += model.cost_of_singleton(current_best_element)
        for e, d in pq:
            if e == current_best_element or model.cost_of_singleton(e) + cur_cost > model.budget:
                not_valid.add(e)
    res = {
        'S': set(solution_set),
        'f(S)': model.objective(solution_set),
        'c(S)': cur_cost,
    }
    return res


def positive_greedy_lazyupdate(model: BaseTask, prob: float = None):
    solution_set = []
    cur_cost = 0.
    N = list(model.ground_set)
    element_densities = [model.density(e, solution_set) for e in N]
    element_densities_sortidx = np.argsort(element_densities)[::-1]
    best_function_value_idx = element_densities_sortidx[0]
    solution_set.append(N[best_function_value_idx])
    lazy_idx = 1
    while len(N):
        if lazy_idx >= len(element_densities_sortidx):
            lazy_idx = len(element_densities_sortidx) - 1
        current_potential_best_element = N[element_densities_sortidx[lazy_idx]]
        current_potential_best_density = model.density(
            current_potential_best_element, solution_set)
        previous_next_best_density = element_densities[element_densities_sortidx[lazy_idx + 1]]
        if current_potential_best_density >= previous_next_best_density:
            lazy_idx += 1
            current_best_element = current_potential_best_element
            current_best_density = current_potential_best_density
        else:
            N = list(set(N) - set(solution_set))
            element_densities = [model.density(e, solution_set) for e in N]
            element_densities_sortidx = np.argsort(element_densities)[::-1]
            best_function_value_idx = element_densities_sortidx[0]
            lazy_idx = 1
            current_best_element = N[best_function_value_idx]
            current_best_density = element_densities[element_densities_sortidx[lazy_idx]]
        max_density = current_best_density
        if max_density < 0.:
            break
        if prob is not None and np.random.uniform(0., 1.) >= prob:
            continue
        else:
            solution_set.append(current_best_element)
            cur_cost += model.cost_of_singleton(current_best_element)
        new_N = []
        for e in N:
            if e == current_best_element:
                continue
            if model.cost_of_singleton(e) + cur_cost > model.budget:
                continue
            new_N.append(e)
        N = new_N
    res = {
        'S': set(solution_set),
        'f(S)': model.objective(solution_set),
        'c(S)': cur_cost,
    }
    return res


def positive_greedy_original(model: BaseTask, prob: float = None):
    """
    Inputs:
    - model: problem instance
    - prob: parameter of Bernoulli distribution, add optimal element with this probability
    """
    sol = set()
    remaining_elements = set(model.ground_set)
    cur_cost = 0.
    while len(remaining_elements):
        u, max_density = None, -1.
        for e in remaining_elements:
            ds = model.density(e, sol)
            if u is None or ds > max_density:
                u, max_density = e, ds
        assert u is not None
        if max_density < 0.:
            break
        if prob is not None and np.random.uniform(0., 1.) >= prob:
            continue
        else:
            sol.add(u)
            cur_cost += model.cost_of_singleton(u)
        remaining_elements.remove(u)
        to_remove = set()
        for v in remaining_elements:
            if model.cost_of_singleton(v) + cur_cost > model.budget:
                to_remove.add(v)
        remaining_elements -= to_remove
    res = {
        'S': sol,
        'f(S)': model.objective(sol),
        'c(S)': cur_cost,
    }
    return res
