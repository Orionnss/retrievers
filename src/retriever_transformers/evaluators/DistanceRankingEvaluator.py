from typing import Dict, List
from .evaluator import RankEvaluator
import numpy as np

class DistanceRankingEvaluator(RankEvaluator):
    def evaluate(self, ranks: List[List[int]]) -> Dict[str, float]:
        rr = []
        accurate = []
        for query_idx, distances in enumerate(ranks):
            good_answer_idx = query_idx
            predicted_answer_idx = np.argmin(distances)
            if good_answer_idx == predicted_answer_idx:
                accurate.append(1)
            else:
                accurate.append(0)
            rank_of_good_answer = np.argsort(distances).tolist().index(good_answer_idx)
            rr.append(1/(rank_of_good_answer+1))
        mrr = np.mean(rr)
        accuracy = np.mean(accurate)
        return {"mrr": mrr, "accuracy": accuracy}