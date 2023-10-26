import pandas as pd
import numpy as np
from typing import List
from typing import Tuple
from typing import Dict
import itertools
from scipy.spatial.distance import cosine

class SimilarItems:
    """Similar items class"""

    @staticmethod
    def similarity(embeddings: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], float]:

        def cosine_similarity(A, B):
            # dot_product = np.dot(A, B)  # скалярное произведение A и B
            # norm_A = np.linalg.norm(A)  # норма (длина) вектора A
            # norm_B = np.linalg.norm(B)  # норма (длина) вектора B
            # similarity = dot_product / (norm_A * norm_B)  # косинусная схожесть
            similarity = 1 - cosine(A, B)
            return similarity

        similarities = [] # список для хранения попарных схожестей
        for key1, value1 in embeddings.items():
            for key2, value2 in embeddings.items():
                if key1 != key2: # исключаем сравнение элемента с самим собой
                    similarity = cosine_similarity(value1, value2) # функция, рассчитывающая схожесть
                    similarities.append((key1, key2, similarity))
        last = {elem[:2]: round(elem[2], 8) for elem in similarities}
        return {tuple(sorted(key)):value for key, value in last.items()}


    @staticmethod
    def knn(
    sim: Dict[Tuple[int, int], float], top: int
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Return closest neighbors for each item.

        Args:
        sim (Dict[Tuple[int, int], float]): <similarity> method output.
        top (int): Number of top neighbors to consider.

        Returns:
        Dict[int, List[Tuple[int, float]]]: Dict with top closest neighbors
        for each item.
        """
        knn_dict = {}
        for key, value in sim.items():
        #if value not in knn_dict:
            knn_dict[key] = value
        new_dict_knn = dict(sorted(knn_dict.items(), key = lambda x: x[1])[::-1])
        d_1 = {}
        for key, value in sim.items():
            if key[0] not in d_1:
                d_1[key[0]] = []
            d_1[key[0]].append((key[1], value))
        dct = dict(sorted(d_1.items(), key = lambda x:x[0]))
        for key, value in sim.items():
            if key[1] not in d_1:
                d_1[key[1]] = []
            d_1[key[1]].append((key[0], value))
    # dct_1 = dict(sorted(d_1.items(), key = lambda x:x[1][1]))
        dct_1 = {key: sorted(value, key=lambda x: x[1], reverse=True) for key, value in d_1.items()}
        return {key: value[:top] for key, value in dct_1.items()}

    @staticmethod
    def knn_price(
        knn_dict: Dict[int, List[Tuple[int, float]]],
        prices: Dict[int, float],
    ) -> Dict[int, float]:
        """Calculate weighted average prices for each item.
    Weights should be positive numbers in [0, 2] interval.

    Args:
        knn_dict (Dict[int, List[Tuple[int, float]]]): <knn> method output.
        prices (Dict[int, float]): Price dict for each item.

    Returns:
        Dict[int, float]: New prices dict, rounded to 2 decimal places.
        """
        knn_price_dict = {}
        for item, neighbors in knn_dict.items():
            weighted_sum = 0.0
            weight_sum = 0.0
            for neighbor, weight in neighbors:
                weight = weight+1
                if 0 <= weight <= 2:
                    weighted_sum += prices[neighbor] * weight
                    weight_sum += weight
            if weight_sum > 0:
                average_price = weighted_sum / weight_sum
                knn_price_dict[item] = round(average_price, 2)
        return knn_price_dict
        # prices = {
        # 3: 10.0,
        # 4: 20.0,
        # 5: 30.0
        # }
    @staticmethod
    def transform(
        embeddings: Dict[int, np.ndarray],
        prices: Dict[int, float],
        top: int,
    ) -> Dict[int, float]:
        """Transforming input embeddings into a dictionary
        with weighted average prices for each item.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.
            prices (Dict[int, float]): Price dict for each item.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, float]: Dict with weighted average prices for each item.
        """
        simm = SimilarItems.similarity(embeddings)
        knn_l = SimilarItems.knn(simm, top)
        knn_pr = SimilarItems.knn_price(knn_l, prices)

        return knn_pr