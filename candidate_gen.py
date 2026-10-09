from similarity import SimilarityCalculator

class CandidateGenerator:
    """Builds a shortlist of items that might be good recommendations."""

    def __init__(self, user_ratings, items, limit=20):
        self.user_ratings = user_ratings   # who rated what
        self.items = items                 # details about each item
        self.limit = limit                 # max number of candidates to return
        self.sim = SimilarityCalculator()   # our similarity tools
        
    def popularity_candidates(self):
        # Count how many users liked each item (rating 4 or 5)
        counts = {}
        for user, ratings in self.user_ratings.items():
            for item, rating in ratings.items():
                if rating >= 4:
                    counts[item] = counts.get(item, 0) + 1

        # Sort: highest count first (ties broken by item name)
        ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))

        # Keep only the item IDs
        result = []
        for item, count in ranked:
            result.append(item)

        return result[:self.limit]

    def content_based_candidates(self, user_id):
        my_ratings = self.user_ratings.get(user_id, {})

        # Step 1: collect the tags of items this user liked
        profile_tags = set()
        for item, rating in my_ratings.items():
            if rating >= 4:
                tags = self.items.get(item, {}).get("tags", set())
                profile_tags = profile_tags | tags

        # Cold start: no liked items -> show popular items instead
        if len(profile_tags) == 0:
            return self.popularity_candidates()

        # Step 2: score every item the user has not rated yet
        scores = {}
        for item, details in self.items.items():
            if item in my_ratings:
                continue                      # skip items already seen
            score = self.sim.jaccard_similarity(profile_tags, details["tags"])
            if score > 0:
                scores[item] = score

        # Step 3: best match first
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        result = []
        for item, score in ranked:
            result.append(item)

        return result[:self.limit]

    def collaborative_candidates(self, user_id):
        my_ratings = self.user_ratings.get(user_id, {})

        # Cold start: no history -> show popular items instead
        if len(my_ratings) == 0:
            return self.popularity_candidates()

        # Step 1: build my vector (one slot per item, 0 if not rated)
        all_items = sorted(self.items.keys())
        my_vector = []
        for item in all_items:
            my_vector.append(my_ratings.get(item, 0))

        scores = {}
        for other_id, other_ratings in self.user_ratings.items():
            if other_id == user_id:
                continue                          # don't compare me with me

            # Step 2: how similar is this other user to me?
            other_vector = []
            for item in all_items:
                other_vector.append(other_ratings.get(item, 0))
            similarity = self.sim.cosine_similarity(my_vector, other_vector)
            if similarity <= 0:
                continue                          # not similar at all, ignore

            # Steps 3 and 4: add up scores for items they liked that I haven't seen
            for item, rating in other_ratings.items():
                if rating >= 4 and item not in my_ratings:
                    scores[item] = scores.get(item, 0) + similarity

        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        result = []
        for item, score in ranked:
            result.append(item)

        return result[:self.limit]

    def hybrid_candidates(self, user_id):
        my_ratings = self.user_ratings.get(user_id, {})

        # Get the shortlist from each strategy
        lists = [
            self.collaborative_candidates(user_id),
            self.content_based_candidates(user_id),
            self.popularity_candidates(),
        ]

        # Find the length of the longest list
        longest = 0
        for one_list in lists:
            if len(one_list) > longest:
                longest = len(one_list)

        # Take 1st from each list, then 2nd from each, and so on
        result = []
        for position in range(longest):
            for one_list in lists:
                if position < len(one_list):
                    item = one_list[position]
                    if item not in result and item not in my_ratings:
                        result.append(item)

        return result[:self.limit]

    
user_ratings = {
    "u1": {"i1": 5, "i2": 3, "i3": 4},
    "u2": {"i1": 4, "i2": 2, "i4": 5},
    "u3": {},
}
items = {
    "i1": {"tags": {"action", "thriller"}, "year": 2024},
    "i2": {"tags": {"comedy"}, "year": 2020},
    "i3": {"tags": {"action", "drama"}, "year": 2023},
    "i4": {"tags": {"comedy", "romance"}, "year": 2022},
    "i5": {"tags": {"action", "comedy"}, "year": 2025},
}

