from similarity import SimilarityCalculator
from candidate_gen import CandidateGenerator
from scorer import RecommendationScorer
from evaluator import RecommendationEvaluator

# ---------- Sample data (plain dictionaries) ----------
user_ratings = {
    "u1": {"i1": 5, "i2": 3, "i3": 4},
    "u2": {"i1": 4, "i2": 2, "i4": 5},
    "u3": {},                          # new user, no history (cold start)
}
items = {
    "i1": {"tags": {"action", "thriller"}, "year": 2024},
    "i2": {"tags": {"comedy"}, "year": 2020},
    "i3": {"tags": {"action", "drama"}, "year": 2023},
    "i4": {"tags": {"comedy", "romance"}, "year": 2022},
    "i5": {"tags": {"action", "comedy"}, "year": 2025},
}


# ---------- Tests for similarity.py ----------
def test_similarity():
    calc = SimilarityCalculator()
    # cosine
    assert round(calc.cosine_similarity([5, 3, 4], [5, 3, 4]), 4) == 1.0
    assert round(calc.cosine_similarity([5, 3, 4], [4, 2, 0]), 4) == 0.8222
    assert calc.cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0
    assert calc.cosine_similarity([], []) == 0.0
    # jaccard
    assert calc.jaccard_similarity({"python", "sql", "ml"}, {"python", "ml", "java"}) == 0.5
    assert calc.jaccard_similarity(set(), set()) == 0.0
    assert calc.jaccard_similarity(set(), {"a"}) == 0.0
    # pearson
    assert round(calc.pearson_correlation([1, 2, 3], [2, 4, 6]), 4) == 1.0
    assert round(calc.pearson_correlation([1, 2, 3], [3, 2, 1]), 4) == -1.0
    assert calc.pearson_correlation([4, 4, 4], [1, 2, 3]) == 0.0
    assert calc.pearson_correlation([], []) == 0.0


# ---------- Tests for candidate_gen.py ----------
def test_candidates():
    gen = CandidateGenerator(user_ratings, items)
    assert gen.popularity_candidates() == ["i1", "i3", "i4"]
    assert gen.content_based_candidates("u1") == ["i5"]
    assert gen.collaborative_candidates("u1") == ["i4"]
    assert gen.hybrid_candidates("u1") == ["i4", "i5"]
    # cold start: new user and unknown user both get popular items
    assert gen.hybrid_candidates("u3") == ["i1", "i3", "i4"]
    assert gen.hybrid_candidates("zzz") == ["i1", "i3", "i4"]
    # limit works
    assert CandidateGenerator(user_ratings, items, limit=1).hybrid_candidates("u3") == ["i1"]
    # empty data
    assert CandidateGenerator({}, {}).popularity_candidates() == []


# ---------- Tests for scorer.py ----------
def test_scorer():
    scorer = RecommendationScorer()
    scorer.add_scorer("relevance", lambda u, i, c: 0.8, 2)
    scorer.add_scorer("popularity", lambda u, i, c: 0.4, 1)
    assert round(scorer.calculate_score("u1", "i1", {}), 4) == 0.6667
    assert RecommendationScorer().calculate_score("u1", "i1", {}) == 0.0
    assert scorer.rank_candidates("u1", [], limit=2) == []

    # ranking order and explanation
    relevance = {"i1": 0.5, "i2": 0.9, "i3": 0.6}
    popularity = {"i1": 0.9, "i2": 0.2, "i3": 0.5}
    ranker = RecommendationScorer()
    ranker.add_scorer("relevance", lambda u, i, c: relevance[i], 2)
    ranker.add_scorer("popularity", lambda u, i, c: popularity[i], 1)
    top = ranker.rank_candidates("u1", ["i1", "i2", "i3"], limit=2)
    assert [r["item"] for r in top] == ["i2", "i1"]
    assert "relevance" in top[0]["reason"]

    # a negative weight is rejected
    try:
        ranker.add_scorer("bad", lambda u, i, c: 1, -1)
        assert False, "negative weight should raise an error"
    except ValueError:
        pass


# ---------- Tests for evaluator.py ----------
def test_evaluator():
    ev = RecommendationEvaluator()
    recs = ["i2", "i5", "i1", "i9"]
    liked = {"i1", "i2", "i3", "i4"}
    assert round(ev.precision_at_k(recs, liked, 3), 4) == 0.6667
    assert round(ev.recall_at_k(recs, liked, 3), 4) == 0.5
    assert round(ev.ndcg_at_k(recs, liked, 3), 4) == 0.7039
    assert ev.precision_at_k([], liked, 3) == 0.0
    assert ev.recall_at_k(recs, set(), 3) == 0.0

    result = ev.evaluate_all(
        {"u1": recs, "u2": ["i3", "i4"], "u3": ["i1"]},
        {"u1": liked, "u2": {"i3"}},        # u3 has no ground truth
        3,
    )
    assert result["users_evaluated"] == 2
    assert round(result["precision_at_k"], 4) == 0.5
    assert round(result["recall_at_k"], 4) == 0.75
    assert round(result["ndcg_at_k"], 4) == 0.852


# ---------- Test that all four parts work together ----------
def test_full_pipeline():
    calc = SimilarityCalculator()
    gen = CandidateGenerator(user_ratings, items)

    # How many users liked each item (used as the popularity factor)
    likes = {}
    for ratings in user_ratings.values():
        for item, rating in ratings.items():
            if rating >= 4:
                likes[item] = likes.get(item, 0) + 1
    most_likes = max(likes.values())

    def popularity_score(user_id, item_id, context):
        return likes.get(item_id, 0) / most_likes

    def recency_score(user_id, item_id, context):
        return (items[item_id]["year"] - 2019) / 6

    def relevance_score(user_id, item_id, context):
        liked_tags = set()
        for item, rating in user_ratings.get(user_id, {}).items():
            if rating >= 4:
                liked_tags = liked_tags | items[item]["tags"]
        return calc.jaccard_similarity(liked_tags, items[item_id]["tags"])

    scorer = RecommendationScorer()
    scorer.add_scorer("relevance", relevance_score, 3)
    scorer.add_scorer("recency", recency_score, 1)
    scorer.add_scorer("popularity", popularity_score, 1)

    all_recs = {}
    for user in ["u1", "u2", "u3"]:
        candidates = gen.hybrid_candidates(user)
        ranked = scorer.rank_candidates(user, candidates, limit=3)
        for r in ranked:
            assert 0.0 <= r["score"] <= 1.0
            assert r["reason"] != ""
        all_recs[user] = [r["item"] for r in ranked]

    assert len(all_recs["u1"]) > 0
    assert len(all_recs["u3"]) > 0          # the new user still gets recommendations

    result = RecommendationEvaluator().evaluate_all(all_recs, {"u1": {"i5"}, "u2": {"i3"}}, 3)
    assert 0.0 <= result["precision_at_k"] <= 1.0


# ---------- Run everything ----------
if __name__ == "__main__":
    tests = [test_similarity, test_candidates, test_scorer, test_evaluator, test_full_pipeline]
    for test in tests:
        test()
        print("PASS:", test.__name__)
    print("All tests passed")
