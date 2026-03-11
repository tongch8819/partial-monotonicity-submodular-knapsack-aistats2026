from positive_modified_greedy import positive_modified_greedy
from positive_greedy_max import positive_greedy_max_lazyupdate as positive_greedy_max
from two_set_enumeration_positive_greedy import two_set_enumeration_positive_greedy
from one_set_enumeration_positive_greedy_max import one_set_enumeration_positive_greedy_max
from sample_greedy import sample_greedy
from linear_query import deterministic_linear_approximation, randomized_linear_approximation
from movie_recommendation import MovieRecommendation
from influence_exploit_marketing import InfluenceAndExplotMarketing
from data_dependent_upperbound import singleton_knapsack_fill
import numpy as np
import json
import argparse
import os
from copy import deepcopy
algo_lst = [
    positive_modified_greedy,
    positive_greedy_max,
    two_set_enumeration_positive_greedy,
    one_set_enumeration_positive_greedy_max,
    sample_greedy,
    deterministic_linear_approximation,
    randomized_linear_approximation,
]
algo_names_lst = [
    "Positive Modified Greedy",
    "Positive Greedy Max",
    "Two Set Enumeration Positive Greedy",
    "One Set Enumeration Positive Greedy Max",
    "Sample Greedy",
    "Deterministic Linear Approximation",
    "Randomized Linear Approximation",
]


def run_for_task(task_handler, output_path, verbose=False):
    num_exp = 5
    monotonicity_ratio_lst = np.linspace(0.1, 1.0, num=num_exp)
    budget_ratio_lst = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    final_result = {}
    for i, br in enumerate(budget_ratio_lst):
        result = {algo_name: [] for algo_name in algo_names_lst}
        for m in monotonicity_ratio_lst:
            llambda = 1 - m / 2
            instance = task_handler(budget_ratio=br, llambda=llambda)
            if verbose:
                print("Monotonicity Ratio:", m)
            upb = singleton_knapsack_fill(instance)
            for algo_name, algo_handler in zip(algo_names_lst, algo_lst):
                res = algo_handler(instance)
                res['AF'] = res['f(S)'] / upb
                if verbose:
                    print(algo_name, res)
                res['S'] = list(res['S'])
                result[algo_name].append(res)
        final_result[br] = deepcopy(result)
    with open(output_path, 'w') as json_file:
        json.dump(final_result, json_file, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="all; movie; influence.")
    if os.path.isdir("result") == False:
        os.mkdir("result")
    args = parser.parse_args()
    if args.task == "all" or args.task == "movie":
        run_for_task(MovieRecommendation,
                     output_path="result/exp_rst_movie.json", verbose=True)
    if args.task == "all" or args.task == "influence":
        run_for_task(InfluenceAndExplotMarketing,
                     output_path="result/exp_rst_influence.json", verbose=True)


if __name__ == "__main__":
    main()
