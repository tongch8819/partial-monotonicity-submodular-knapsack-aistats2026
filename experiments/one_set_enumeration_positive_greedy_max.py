from set_enumeration import set_enumeration
from base_task import BaseTask
from positive_greedy_max import positive_greedy_max_lazyupdate as positive_greedy_max

def one_set_enumeration_positive_greedy_max(model: BaseTask):
    return set_enumeration(model, alg_handler=positive_greedy_max, num_initial_elements=1, test = True)


