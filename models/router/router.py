"""
Query Router for SatQuery AI.

Routes a user query to the appropriate downstream task:
- vqa
- grounding
- change_detection
- fusion
"""

from typing import Dict

from query_classifier import classify_query


class QueryRouter:
    """Routes queries based on their classified task."""

    # Minimum confidence required for a normal route.
    CONFIDENCE_THRESHOLD = 0.60

    def __init__(self):
        """Initialize the query router."""
        pass

    def route_query(self, query: str) -> Dict:
        """
        Classify and route a user query.

        Args:
            query: Natural-language remote-sensing query.

        Returns:
            Dictionary containing routing information.
        """

        # ---------------------------------------------------------
        # 1. Validate input
        # ---------------------------------------------------------

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        query = query.strip()

        # ---------------------------------------------------------
        # 2. Handle empty query
        # ---------------------------------------------------------

        if not query:
            return {
                "query": query,
                "task": "vqa",
                "confidence": 0.0,
                "module": "geochat",
                "status": "invalid_query",
                "message": "Query cannot be empty."
            }

        # ---------------------------------------------------------
        # 3. Classify query
        # ---------------------------------------------------------

        result = classify_query(query)

        task = result["task"]
        confidence = result["confidence"]

        # ---------------------------------------------------------
        # 4. Map task to downstream module
        # ---------------------------------------------------------

        module_map = {
            "vqa": "geochat",
            "grounding": "sam_clip",
            "change_detection": "change_detection",
            "fusion": "fusion_pipeline",
        }

        module = module_map.get(task)

        # Safety fallback
        if module is None:
            return {
                "query": query,
                "task": "vqa",
                "confidence": 0.0,
                "module": "geochat",
                "status": "routing_error",
                "message": "Unknown task detected."
            }

        # ---------------------------------------------------------
        # 5. Check confidence
        # ---------------------------------------------------------

        if confidence < self.CONFIDENCE_THRESHOLD:
            return {
                "query": query,
                "task": task,
                "confidence": confidence,
                "module": module,
                "status": "low_confidence",
                "message": (
                    "The query could not be classified with "
                    "high enough confidence."
                )
            }

        # ---------------------------------------------------------
        # 6. Successful routing
        # ---------------------------------------------------------

        return {
            "query": query,
            "task": task,
            "confidence": confidence,
            "module": module,
            "status": "success",
            "message": "Query routed successfully."
        }


# -------------------------------------------------------------
# Reusable router instance
# -------------------------------------------------------------

router = QueryRouter()


def route_query(query: str) -> Dict:
    """
    Convenience function for other modules.

    Example:
        route_query("Where are the buildings?")
    """

    return router.route_query(query)


# -------------------------------------------------------------
# Manual test
# -------------------------------------------------------------

if __name__ == "__main__":

    test_queries = [
        "Where are the buildings?",
        "What is visible in the image?",
        "What changed between the two images?",
        "Use optical and SAR imagery",
        "",
    ]

    print("\nSatQuery AI Router Test\n")

    for query in test_queries:

        try:
            result = route_query(query)

            print(f"Query: {query}")
            print(f"Task: {result['task']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Module: {result['module']}")
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")

        except Exception as error:
            print(f"Query: {query}")
            print(f"ERROR: {error}")

        print("-" * 50)