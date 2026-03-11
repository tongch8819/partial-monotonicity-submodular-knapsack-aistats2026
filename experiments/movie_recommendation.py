from base_task import BaseTask
import numpy as np
import os
from typing import Iterable, List


class MovieRecommendation(BaseTask):
    def __init__(self, budget_ratio: float, k: int = None, n: int = None, sim_type: str = "cosine", llambda: float = 0.5):
        """
        Inputs:
        - budget_ratio: ratio between budget and total cost
        - k: number of users
        - n: number of movies
        The objective is non-negative and non-monotone.
        Use precomputed weights for movie recommendation
        """
        self.M = self.load_matrix(
            "dataset/movie/user_by_movies_small_rating.npy")
        if k is not None and n is not None:
            self.M = self.M[:k, :n]
        if n is not None:
            self.M = self.M[:, :n]
        if k is not None:
            self.M = self.M[:k, :]
        self.num_users, self.num_movies = self.M.shape
        self.movies = [i for i in range(self.num_movies)]
        self.similarity_type = sim_type
        self.weights = np.zeros(shape=(self.num_movies, self.num_movies))
        for i in range(self.num_movies):
            for j in range(self.num_movies):
                self.weights[i, j] = self._similarity(i, j)
        avg_ratings = np.average(self.M, axis=0)
        assert min(avg_ratings) >= 0 and max(
            avg_ratings) <= 10, "Average rating should lie in [0, 10]"
        self.costs_obj = 10 - avg_ratings
        self.llambda = llambda
        assert 0. <= self.llambda <= 1.
        self.b = budget_ratio * sum(self.costs_obj)

    @property
    def ground_set(self):
        return self.movies

    @ground_set.setter
    def ground_set(self, v):
        self.movies = v

    def load_matrix(self, path: str):
        if not os.path.isfile(path):
            raise OSError("File *.npy does not exist.")
        return np.load(path)

    def _similarity(self, u, v):
        """
        Inputs:
        - u: some movie
        - v: some movie
        """
        u_vec, v_vec = self.M[:, u], self.M[:, v]
        if self.similarity_type == 'inner':
            return np.dot(u_vec, v_vec)
        elif self.similarity_type == 'cosine':
            return np.dot(u_vec, v_vec) / (np.linalg.norm(u_vec) * np.linalg.norm(v_vec))
        elif self.similarity_type == 'exp':
            euclidean_dist = np.linalg.norm(u_vec - v_vec)
            llambda = 2.
            return np.exp(- llambda * euclidean_dist)
        else:
            raise ValueError("Unsupported similarity type.")

    def objective(self, S: List[int]):
        """
        Inputs:
        - S: solution set
        - llambda: coefficient which lies in [0,1]
        """
        first = 0.
        second = 0.
        if not isinstance(S, Iterable):
            S = [S]
        S = set(S)
        for v in S:
            for u in self.ground_set:
                s_uv = self.weights[u, v]
                first += s_uv
                if u in S:
                    second += s_uv
        return first - self.llambda * second


def main():
    model = MovieRecommendation(n=10000, budget_ratio=0.15, llambda=0.75)
    print("V = ", len(model.ground_set))
    S = [0, 1, 2, 3, 4]
    print("S =", S)
    print("f(S) =", model.objective(S))
    print("c(S) =", model.cost_of_set(S))


if __name__ == "__main__":
    main()
