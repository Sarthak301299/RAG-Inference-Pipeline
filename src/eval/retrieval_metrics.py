import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of the top-k retrieved ids that are relevant.

    Returns 0.0 if k <= 0 or no ids were retrieved within the top-k window.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of all relevant ids that appear within the top-k retrieved ids.

    Returns 0.0 if there are no relevant ids to find (avoids div-by-zero).
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if not relevant_ids:
        return 0.0

    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Reciprocal rank of the first relevant id in the retrieved list.

    Returns 0.0 if no relevant id is found anywhere in retrieved_ids.
    """
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def average_metric(values: list[float]) -> float:
    """Simple mean helper used to aggregate per-query metric values."""
    if not values:
        return 0.0
    return sum(values) / len(values)
