import math


class SimilarityCalculator:
    """Measures how similar two users, items or skills are."""

    def cosine_similarity(self, vec1, vec2):
        # Edge case: empty lists or lists of different size
        if len(vec1) == 0 or len(vec2) == 0 or len(vec1) != len(vec2):
            return 0.0

        # Top part: multiply matching numbers and add them up
        dot = 0
        for i in range(len(vec1)):
            dot = dot + vec1[i] * vec2[i]

        # Bottom part: length of each vector
        length1 = math.sqrt(sum(x * x for x in vec1))
        length2 = math.sqrt(sum(x * x for x in vec2))

        # Edge case: a zero vector like [0, 0, 0] would divide by zero
        if length1 == 0 or length2 == 0:
            return 0.0

        return dot / (length1 * length2)

    def jaccard_similarity(self, set1, set2):
        # Convert to sets so duplicates are ignored
        set1 = set(set1)
        set2 = set(set2)

        # Edge case: both empty -> nothing to compare
        if len(set1) == 0 and len(set2) == 0:
            return 0.0

        shared = set1 & set2      # items in both
        combined = set1 | set2    # items in either

        return len(shared) / len(combined)

    def pearson_correlation(self, ratings1, ratings2):
        # Edge case: empty, different sizes, or fewer than 2 ratings
        if len(ratings1) != len(ratings2) or len(ratings1) < 2:
            return 0.0

        # Step 1: average rating of each person
        mean1 = sum(ratings1) / len(ratings1)
        mean2 = sum(ratings2) / len(ratings2)

        # Step 2: compare each rating to that person's average
        top = 0
        spread1 = 0
        spread2 = 0
        for i in range(len(ratings1)):
            diff1 = ratings1[i] - mean1
            diff2 = ratings2[i] - mean2
            top = top + diff1 * diff2
            spread1 = spread1 + diff1 * diff1
            spread2 = spread2 + diff2 * diff2

        # Edge case: someone gave the same rating to everything
        if spread1 == 0 or spread2 == 0:
            return 0.0

        # Step 3: combine
        return top / math.sqrt(spread1 * spread2)

