from abc import ABC
from typing import Iterable, List, Set
from copy import deepcopy


class BaseTask(ABC):

    def __init__(self):
        self.total_curvature = None

    @property
    def budget(self):
        return self.b

    @budget.setter
    def budget(self, v):
        self.b = v

    def objective(self):
        pass

    def cost_of_set(self, S: List[int]):
        if not isinstance(S, Iterable):
            S = [S]
        return sum(self.costs_obj[x] for x in S)

    def cost_of_singleton(self, singleton: int):
        assert singleton < len(
            self.costs_obj), "Singleton: {}".format(singleton)
        return self.costs_obj[singleton]

    def marginal_gain(self, single: int, base: List[int]):
        if len(base) == 0:
            base2 = [single]
        else:
            base = list(base)
            base2 = list(set(base + [single]))
        fS1 = self.objective(base)
        fS2 = self.objective(base2)
        res = fS2 - fS1
        return res

    def density(self, single: int, base: List[int]):
        mg = self.marginal_gain(single, base)
        cost = self.costs_obj[single]
        return (mg * 100) / (cost * 100)

    def cutout_marginal_gain(self, singleton: int, base: Set[int]):
        """
        return f(singleton | base - singleton)
        """
        if type(base) is not set:
            base = set(base)
        assert singleton in base
        
        base2 = deepcopy(base)
        base2.remove(singleton)  
        fS2, fS1 = self.objective(base), self.objective(base2)
        res = fS2 - fS1
        assert res >= 0., f"f({base}) - f({base2}) = {fS2:.2f} - {fS1:.2f}\n{base - base2}"
        return res

    def cutout_density(self, singleton: int, base: Set[int]):
        return self.cutout_marginal_gain(singleton, base) / self.cost_of_singleton(singleton)

    def gen_total_curvature(self):
        if self.total_curvature is not None:
            return self.total_curvature

        V = self.ground_set
        min_ratio = 1
        for e in self.ground_set:
            nume = self.cutout_marginal_gain(e, V)
            denom = self.objective([e])
            assert denom >= nume
            ratio = nume / (denom + 1e-7)
            min_ratio = min(min_ratio, ratio)
        self.total_curvature = 1 - min_ratio
        return self.total_curvature
