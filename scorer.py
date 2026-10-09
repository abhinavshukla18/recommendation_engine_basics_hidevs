class RecommendationScorer:
    """Scores items using several weighted scoring functions."""

    def __init__(self):
        self.scorers = {}    # name -> {"function": ..., "weight": ...}

    def add_scorer(self, name, function, weight):
        if weight < 0:
            raise ValueError("weight cannot be negative")
        self.scorers[name] = {"function": function, "weight": weight}

    def calculate_score(self, user_id, item_id, context=None):
        if context is None:
            context = {}

        # Edge case: no scoring functions added yet
        if len(self.scorers) == 0:
            return 0.0

        total = 0
        total_weight = 0
        for name, info in self.scorers.items():
            score = info["function"](user_id, item_id, context)
            score = max(0.0, min(1.0, score))     # keep each score between 0 and 1
            total = total + score * info["weight"]
            total_weight = total_weight + info["weight"]

        # Edge case: all weights are 0
        if total_weight == 0:
            return 0.0

        return total / total_weight

    def explain(self, user_id, item_id, context=None):
        if context is None:
            context = {}

        best_name = None
        best_value = -1
        best_score = 0
        for name, info in self.scorers.items():
            score = info["function"](user_id, item_id, context)
            score = max(0.0, min(1.0, score))
            value = score * info["weight"]       # how much this factor contributed
            if value > best_value:
                best_value = value
                best_name = name
                best_score = score

        if best_name is None:
            return "No scoring factors were set"
        return "Recommended mainly because of " + best_name + " (score " + str(round(best_score, 2)) + ")"

    def rank_candidates(self, user_id, candidates, limit=10, context=None):
        # Edge case: nothing to rank
        if not candidates:
            return []

        results = []
        for item in candidates:
            score = self.calculate_score(user_id, item, context)
            reason = self.explain(user_id, item, context)
            results.append({"item": item, "score": score, "reason": reason})

        # Highest score first (ties broken by item name)
        results.sort(key=lambda r: (-r["score"], r["item"]))

        return results[:limit]

relevance = {"i1": 0.5, "i2": 0.9, "i3": 0.6}
popularity = {"i1": 0.9, "i2": 0.2, "i3": 0.5}

scorer = RecommendationScorer()
scorer.add_scorer("relevance", lambda u, i, c: relevance[i], 2)
scorer.add_scorer("popularity", lambda u, i, c: popularity[i], 1)

top = scorer.rank_candidates("u1", ["i1", "i2", "i3"], limit=2)
for r in top:
    print(r["item"], round(r["score"], 4), r["reason"])

print(scorer.rank_candidates("u1", [], limit=2))

