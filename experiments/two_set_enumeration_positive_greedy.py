from set_enumeration import set_enumeration
from base_task import BaseTask
from positive_greedy import positive_greedy_lazyupdate2 as positive_greedy


def two_set_enumeration_positive_greedy(model: BaseTask):
    return set_enumeration(model, alg_handler=positive_greedy, num_initial_elements=2, test=True)
