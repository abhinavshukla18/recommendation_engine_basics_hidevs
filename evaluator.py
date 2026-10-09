import math

class RecommendationEvaluator:
    """Checks how good a list of recommendations is."""

    def precision_at_k(self, recommendations, relevant_items, k):
        # Edge cases: bad k, nothing recommended, or no ground truth
        if k <= 0 or not recommendations or not relevant_items:
            return 0.0

        relevant = set(relevant_items)
        hits = 0
        for item in recommendations[:k]:       # only look at the top k
            if item in relevant:
                hits = hits + 1

        return hits / k

    def recall_at_k(self, recommendations, relevant_items, k):
        if k <= 0 or not recommendations or not relevant_items:
            return 0.0

        relevant = set(relevant_items)
        hits = 0
        for item in recommendations[:k]:
            if item in relevant:
                hits = hits + 1

        return hits / len(relevant)

    def ndcg_at_k(self, recommendations, relevant_items, k):
        if k <= 0 or not recommendations or not relevant_items:
            return 0.0

        relevant = set(relevant_items)

        # DCG: reward each hit, hits near the top are worth more
        dcg = 0
        position = 1
        for item in recommendations[:k]:
            if item in relevant:
                dcg = dcg + 1 / math.log2(position + 1)
            position = position + 1

        # IDCG: the best possible score (all liked items at the top)
        idcg = 0
        best_hits = min(len(relevant), k)
        for position in range(1, best_hits + 1):
            idcg = idcg + 1 / math.log2(position + 1)

        return dcg / idcg

    def evaluate_all(self, recommendations_dict, ground_truth_dict, k):
        totals = {"precision_at_k": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0}
        users_evaluated = 0

        for user_id, recs in recommendations_dict.items():
            relevant = ground_truth_dict.get(user_id)
            if not relevant:
                continue                      # no ground truth -> skip this user

            totals["precision_at_k"] = totals["precision_at_k"] + self.precision_at_k(recs, relevant, k)
            totals["recall_at_k"] = totals["recall_at_k"] + self.recall_at_k(recs, relevant, k)
            totals["ndcg_at_k"] = totals["ndcg_at_k"] + self.ndcg_at_k(recs, relevant, k)
            users_evaluated = users_evaluated + 1

        # Average over the users we could evaluate
        if users_evaluated > 0:
            for name in ["precision_at_k", "recall_at_k", "ndcg_at_k"]:
                totals[name] = totals[name] / users_evaluated

        totals["users_evaluated"] = users_evaluated
        return totals



