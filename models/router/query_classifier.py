"""
Query Classifier for SatQuery AI.

Classifies remote-sensing queries into:
- vqa
- grounding
- change_detection
- fusion
"""

import re
from typing import Dict


class QueryClassifier:
    """Classifies natural-language queries into remote-sensing task types."""

    TASK_KEYWORDS = {
        "grounding": [
            "where",
            "locate",
            "located",
            "find",
            "show",
            "location",
            "identify",
            "point out",
            "mark",
            "highlight",
            "detect",
        ],

        "change_detection": [
            "changed",
            "change",
            "different",
            "difference",
            "before and after",
            "before/after",
            "compare",
            "comparison",
            "increased",
            "decreased",
            "increase",
            "decrease",
            "new",
            "removed",
            "disappeared",
            "growth",
        ],

        "fusion": [
            "sar",
            "sentinel-1",
            "sentinel 1",
            "sentinel-2 and sar",
            "optical and sar",
            "multispectral and sar",
            "optical sar",
            "fused",
            "fusion",
        ],

        "vqa": [
            "what",
            "describe",
            "how much",
            "how many",
            "which",
            "is there",
            "are there",
            "does the image",
        ],
    }

    def __init__(self):
        """Initialize the query classifier."""
        pass

    def normalize_query(self, query: str) -> str:
        """
        Clean and normalize the user's query.

        Example:
            "  WHERE are the buildings??? "
            -> "where are the buildings"
        """

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        query = query.lower().strip()

        # Remove unnecessary punctuation
        query = re.sub(r"[^\w\s/-]", "", query)

        # Replace multiple spaces with one
        query = re.sub(r"\s+", " ", query)

        return query

    def classify(self, query: str) -> Dict:
        """
        Classify a remote-sensing query.

        Returns:
            {
                "task": "grounding",
                "confidence": 0.90
            }
        """

        query = self.normalize_query(query)

        # Empty query
        if not query:
            return {
                "task": "vqa",
                "confidence": 0.0
            }

        scores = {
            "grounding": 0,
            "change_detection": 0,
            "fusion": 0,
            "vqa": 0,
        }

        # Count keyword matches
        for task, keywords in self.TASK_KEYWORDS.items():
            for keyword in keywords:

                # Match complete words/phrases instead of random substrings
                pattern = r"\b" + re.escape(keyword) + r"\b"

                if re.search(pattern, query):
                    scores[task] += 1

        # ---------------------------------------------------------
        # Priority rules for common ambiguous queries
        # ---------------------------------------------------------

        # Change detection should have highest priority
        # when the query explicitly asks about changes/comparison.
        if scores["change_detection"] > 0:
            best_task = "change_detection"

        # Grounding should win when the user asks where to
        # locate/find/identify/mark something.
        elif scores["grounding"] > 0:
            best_task = "grounding"

        # Fusion when the query explicitly mentions SAR,
        # optical + SAR, Sentinel-1 etc.
        elif scores["fusion"] > 0:
            best_task = "fusion"

        # Otherwise use VQA.
        elif scores["vqa"] > 0:
            best_task = "vqa"

        # No known keyword → default VQA
        else:
            return {
                "task": "vqa",
                "confidence": 0.50
            }

        best_score = scores[best_task]

        # Calculate confidence
        confidence = min(
            0.95,
            0.70 + (best_score - 1) * 0.10
        )

        return {
            "task": best_task,
            "confidence": round(confidence, 2)
        }


# Reusable classifier instance
classifier = QueryClassifier()


def classify_query(query: str) -> Dict:
    """
    Convenience function used by other modules.

    Example:
        classify_query("Where are the buildings?")
    """

    return classifier.classify(query)