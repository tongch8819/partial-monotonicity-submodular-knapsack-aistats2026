"""
This script implements the approximate algorithms from paper Linear Query Approximation Algorithms for Non-monotone Submodular
Maximization under Knapsack 
"""
from submodular.application.base_model import BaseModel
from ordered_set import OrderedSet
import math
from itertools import accumulate
from bisect import bisect
from collections.abc import Iterable
import random


def _linear_approximation(objective_function, ground_set, budget, cost_function, marginal_gain_function):
    B = budget
    V_1 = OrderedSet(filter(lambda e: cost_function(e) <= B / 2, ground_set))
    X, Y = OrderedSet(), OrderedSet()
    e_max = max(ground_set, key=objective_function, default=None)
    for e in V_1:
        candidate_set = []
        if marginal_gain_function(e, X) / cost_function(e) >= objective_function(X) / B:
            candidate_set.append(X)
        if marginal_gain_function(e, Y) / cost_function(e) >= objective_function(Y) / B:
            candidate_set.append(Y)
        if len(candidate_set) == 0:
            continue
        Z = max(candidate_set, key=lambda s: marginal_gain_function(
            e, s) / cost_function(e))
        Z.add(e)
    reversed_X = OrderedSet(reversed(X))
    reversed_Y = OrderedSet(reversed(Y))
    acc_cost_X = list(accumulate([cost_function(e)
                      for e in reversed_X], initial=0))
    acc_cost_Y = list(accumulate([cost_function(e)
                      for e in reversed_Y], initial=0))
    X_prime_index = bisect(acc_cost_X, B)
    Y_prime_index = bisect(acc_cost_Y, B)
    X_prime = reversed_X[:X_prime_index]
    Y_prime = reversed_Y[:Y_prime_index]
    candidate_sol_seq = [X_prime, Y_prime, OrderedSet([e_max])]
    sol = max(candidate_sol_seq, key=objective_function)
    return sol


def _linear_approximation_randomized(objective_function, ground_set, budget, cost_function, marginal_gain_function, probability=0.5, alpha=0.5):
    """
    LAR algorithm for non-monotone submodular maximization with knapsack constraints.
    """
    B = budget
    e_max = max(ground_set, key=objective_function, default=None)
    V_1 = OrderedSet(filter(lambda e: cost_function(e)
                     <= budget / 2, ground_set))
    V_p = OrderedSet(e for e in V_1 if random.random() <= probability)
    S = OrderedSet()
    for e in V_p:
        if marginal_gain_function(e, S) / cost_function(e) >= alpha * objective_function(S) / budget:
            S.add(e)
    reversed_S = OrderedSet(reversed(S))
    acc_cost_S = list(accumulate([cost_function(e)
                      for e in reversed_S], initial=0))
    S_prime_index = bisect(acc_cost_S, B)
    S_prime = reversed_S[:S_prime_index]
    candidate_sol_seq = [S_prime, OrderedSet([e_max])]
    return max(candidate_sol_seq, key=objective_function)


def deterministic_linear_approximation_multiple_parameter(objective_function, marginal_gain_function,
                                                          ground_set: OrderedSet, budget, cost_function, epsilon):
    """
    DLA algorithm for non-monotone submodular maximization with knapsack constraints.
    """
    B = budget
    V = ground_set
    S_prime = _linear_approximation(
        objective_function, ground_set, budget, cost_function, marginal_gain_function)
    Gamma = objective_function(S_prime)
    epsilon_prime = epsilon / 14
    theta = 19 * Gamma / (6 * epsilon_prime * B)
    X, cost_X = OrderedSet(), 0.
    Y, cost_Y = OrderedSet(), 0.
    while theta >= Gamma * (1 - epsilon_prime) / (6 * B):
        outside_elements = V.difference(X.union(Y))
        for e in outside_elements:
            candidate_set = []
            if cost_X + cost_function(e) <= B:
                candidate_set.append(X)
            if cost_Y + cost_function(e) <= B:
                candidate_set.append(Y)
            threshold_candidate_set = list(
                filter(lambda x: marginal_gain_function(e, x) >= theta, candidate_set))
            if len(threshold_candidate_set) == 0:
                continue
            elif len(threshold_candidate_set) == 1:
                if threshold_candidate_set[0] == X:
                    X.add(e)
                    cost_X += cost_function(e)
                else:
                    Y.add(e)
                    cost_Y += cost_function(e)
            else:
                if marginal_gain_function(e, X) >= marginal_gain_function(e, Y):
                    X.add(e)
                    cost_X += cost_function(e)
                else:
                    Y.add(e)
                    cost_Y += cost_function(e)
        theta *= (1 - epsilon_prime)
    Delta = math.ceil(math.log(1 / epsilon_prime) / epsilon_prime)
    acc_cost_X = list(accumulate([cost_function(e) for e in X], initial=0))
    acc_cost_Y = list(accumulate([cost_function(e) for e in Y], initial=0))
    X_seq, Y_seq = [], []
    for l in range(Delta + 1):
        criterion = epsilon_prime * B * ((1 + epsilon_prime) ** l)
        X_prime_l_index = bisect(acc_cost_X, criterion)
        Y_prime_l_index = bisect(acc_cost_Y, criterion)
        X_prime_l = X[:X_prime_l_index]
        Y_prime_l = Y[:Y_prime_l_index]
        e_X_optimal = max((e for e in ground_set if cost_function(X_prime_l.union(OrderedSet(
            [e]))) <= B), key=lambda e: objective_function(X.union(OrderedSet([e]))), default=None)
        e_Y_optimal = max((e for e in ground_set if cost_function(Y_prime_l.union(OrderedSet(
            [e]))) <= B), key=lambda e: objective_function(X.union(OrderedSet([e]))), default=None)
        X_seq.append(X_prime_l.union(OrderedSet([e_X_optimal])))
        Y_seq.append(Y_prime_l.union(OrderedSet([e_Y_optimal])))
    candidate_solution_seq = []
    candidate_solution_seq.append(S_prime)
    candidate_solution_seq.append(X)
    candidate_solution_seq.append(Y)
    candidate_solution_seq.extend(X_seq)
    candidate_solution_seq.extend(Y_seq)
    optimal_solution = max(candidate_solution_seq, key=objective_function)
    return optimal_solution


def randomized_linear_approximation_multiple_parameter(objective_function, marginal_gain_function,
                                                       ground_set: OrderedSet, budget, cost_function, epsilon):
    """
    RLA algorithm for non-monotone submodular maximization with knapsack constraints.
    """
    B = budget
    specified_probability = math.sqrt(2) - 1
    specified_alpha = math.sqrt(2 + 2 * math.sqrt(2))
    S_prime = _linear_approximation_randomized(
        objective_function, ground_set, budget, cost_function, marginal_gain_function, probability=specified_probability, alpha=specified_alpha)
    S_j, U = OrderedSet(), OrderedSet()
    S_seq = []
    j = 0
    Gamma = objective_function(S_prime)
    epsilon_prime = epsilon / 10
    theta = 16.034 * Gamma / (4 * epsilon_prime * B)
    while theta >= Gamma * (1 - epsilon_prime) / (4 * B):
        iterative_elements = ground_set.copy()
        for e in iterative_elements:
            if marginal_gain_function(e, S_j) >= theta and cost_function(S_j.union(OrderedSet([e]))) <= B:
                U.add(e)
                if random.random() <= 0.5:
                    S_j.add(e)
                j += 1
                S_seq.append(S_j.copy())
        theta *= (1 - epsilon_prime)
    acc_cost_S_j = [cost_function(e) for e in [OrderedSet()] + S_seq]
    S_l_seq = []
    num_levels = math.ceil(math.log(1 / epsilon_prime) / epsilon_prime) + 1
    for l in range(num_levels):
        threshold = epsilon_prime * B * ((1 + epsilon_prime) ** l)
        S_prime_l_index = bisect(acc_cost_S_j, threshold)
        if S_prime_l_index <= 0:
            S_prime_l = OrderedSet()
        elif S_prime_l_index - 1 < len(S_seq):
            S_prime_l = S_seq[S_prime_l_index - 1]
        else:
            S_prime_l = S_seq[-1]
        e_l_max = max((e for e in ground_set if cost_function(S_prime_l.union(OrderedSet(
            [e]))) <= B), key=lambda e: objective_function(S_prime_l.union(OrderedSet([e]))), default=None)
        S_l = S_prime_l.union(OrderedSet([e_l_max]))
        S_l_seq.append(S_l)
    candidate_solution_seq = [S_prime, S_j] + S_l_seq
    optimal_solution = max(candidate_solution_seq, key=objective_function)
    return optimal_solution


def deterministic_linear_approximation(model: BaseModel, epsilon: float = 0.1):
    """Epsilon is set as 0.1 from paper: Linear Query Approximation Algorithms for Non-monotone Submodular
Maximization under Knapsack """
    objective_function = model.objective
    marginal_gain_function = model.marginal_gain
    ground_set = OrderedSet(model.ground_set)
    budget = model.budget
    cost_function = model.cost_of_set
    optimal_solution = deterministic_linear_approximation_multiple_parameter(
        objective_function, marginal_gain_function, ground_set, budget, cost_function, epsilon)
    res = {
        'S': optimal_solution,
        'f(S)': objective_function(optimal_solution),
        'cost(S)': cost_function(optimal_solution),
    }
    return res


def randomized_linear_approximation(model: BaseModel, epsilon: float = 0.1):
    """Epsilon is set as 0.1 from paper: Linear Query Approximation Algorithms for Non-monotone Submodular
Maximization under Knapsack """
    objective_function = model.objective
    marginal_gain_function = model.marginal_gain
    ground_set = OrderedSet(model.ground_set)
    budget = model.budget
    cost_function = model.cost_of_set
    optimal_solution = randomized_linear_approximation_multiple_parameter(
        objective_function, marginal_gain_function, ground_set, budget, cost_function, epsilon)
    res = {
        'S': optimal_solution,
        'f(S)': objective_function(optimal_solution),
        'cost(S)': cost_function(optimal_solution),
    }
    return res


def main():
    def objective_function(S):
        if not isinstance(S, Iterable) or isinstance(S, (str, bytes)):
            return S
        return sum(S)

    def cost_function(e):
        if isinstance(e, Iterable) and not isinstance(e, (str, bytes)):
            return sum(e)
        return e

    def marginal_gain_function(e, S):
        return objective_function(S.union(OrderedSet([e]))) - objective_function(S)
    ground_set = OrderedSet(range(1, 11))
    budget = 15
    epsilon = 0.1
    deterministic_solution = deterministic_linear_approximation_multiple_parameter(
        objective_function, marginal_gain_function, ground_set, budget, cost_function, epsilon)
    print("Optimal solution (deterministic):", deterministic_solution,
          objective_function(deterministic_solution))
    randomized_solution = randomized_linear_approximation_multiple_parameter(
        objective_function, marginal_gain_function, ground_set, budget, cost_function, epsilon)
    print("Optimal solution (randomized):", randomized_solution,
          objective_function(randomized_solution))


if __name__ == "__main__":
    main()
