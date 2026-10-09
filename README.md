# Recommendation Engine Core Components

A small recommendation engine built with plain Python (no databases, no external libraries).
It shows, in a simple way, how platforms like Netflix and Amazon suggest items.

## How the parts fit together

```
similarity.py  ->  candidate_gen.py  ->  scorer.py  ->  evaluator.py
 (how alike?)      (build shortlist)     (rank it)      (is it good?)
```

## Components

### 1. similarity.py - `SimilarityCalculator`
Measures how alike two users, items or skills are.
- `cosine_similarity(vec1, vec2)` - compares two lists of numbers (e.g. rating vectors). Returns 0 to 1.
- `jaccard_similarity(set1, set2)` - compares two sets (e.g. tags, skills) by overlap. Returns 0 to 1.
- `pearson_correlation(ratings1, ratings2)` - compares rating patterns. Returns -1 to 1.

Edge cases: empty inputs, zero vectors, different lengths and constant ratings all return 0.0 instead of crashing.

### 2. candidate_gen.py - `CandidateGenerator`
Builds a shortlist (default 20 items) of items worth considering. Created with `CandidateGenerator(user_ratings, items, limit=20)`.
- `popularity_candidates()` - items liked (rated 4 or 5) by the most users.
- `content_based_candidates(user_id)` - unseen items whose tags match items the user liked (uses Jaccard).
- `collaborative_candidates(user_id)` - unseen items liked by users with similar ratings (uses cosine).
- `hybrid_candidates(user_id)` - mixes the three lists, without duplicates and without items the user already rated.

Cold start: a user with no history (or an unknown user) gets popular items.

### 3. scorer.py - `RecommendationScorer`
Scores shortlisted items and picks the best.
- `add_scorer(name, function, weight)` - register a scoring function (returns 0 to 1) with a weight.
- `calculate_score(user_id, item_id, context)` - weighted average of all scoring functions, between 0 and 1.
- `explain(user_id, item_id, context)` - names the factor that contributed most.
- `rank_candidates(user_id, candidates, limit, context)` - returns the top items as `{"item", "score", "reason"}`.

### 4. evaluator.py - `RecommendationEvaluator`
Checks whether recommendations are good, by comparing them with items the user actually liked.
- `precision_at_k` - share of the top k that were relevant.
- `recall_at_k` - share of the relevant items that were found in the top k.
- `ndcg_at_k` - like precision, but rewards putting relevant items higher up.
- `evaluate_all(recommendations_dict, ground_truth_dict, k)` - averages all three metrics over users; users without ground truth are skipped.

## Sample data format
```python
user_ratings = {"u1": {"i1": 5, "i2": 3}, "u3": {}}                 # user -> {item: rating 1-5}
items = {"i1": {"tags": {"action"}, "year": 2024}}                  # item -> details
```

## Running the tests
```
python test.py
```
Every test prints PASS and the last line is `All tests passed`. `test.py` also runs a full pipeline test:
candidates -> scoring -> ranking -> evaluation.
