from typing import List, Dict
import numpy as np
from .evaluator import RankEvaluator

class ScoreRankingEvaluator(RankEvaluator):
    def evaluate(self, ranks: List[List[int]]) -> Dict[str, float]:
        rr = []
        accurate = []
        for query_idx, distances in enumerate(ranks):
            good_answer_idx = query_idx
            predicted_answer_idx = np.argmax(distances)
            if good_answer_idx == predicted_answer_idx:
                accurate.append(1)
            else:
                accurate.append(0)
            rank_of_good_answer = np.argsort(distances).tolist()[::-1].index(good_answer_idx)
            rr.append(1/(rank_of_good_answer+1))
        mrr = np.mean(rr)
        accuracy = np.mean(accurate)
        return {"mrr": mrr, "accuracy": accuracy}
            