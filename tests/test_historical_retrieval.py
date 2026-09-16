import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from historical_retrieval import HistoricalRetriever  # noqa: E402


@pytest.fixture(scope="module")
def retriever():
    return HistoricalRetriever.from_csv()


def test_retrieval_returns_historical_evidence(retriever):
    hits = retriever.retrieve("spotify keeps crashing after an update", top_k=3)
    assert len(hits) == 3
    assert all(hit["customer_message"] for hit in hits)
    assert all(hit["brand_response"] for hit in hits)
    assert all(0.0 <= hit["similarity"] <= 1.0 for hit in hits)


def test_retrieval_preserves_historical_response_and_identity(retriever):
    hit = retriever.retrieve("my downloaded songs disappeared", top_k=1)[0]
    assert hit["interaction_id"]
    assert hit["brand_tweet_id"]
    assert isinstance(hit["brand_response"], str)


def test_intent_scoping_uses_available_v2_1_metadata(retriever):
    hits = retriever.retrieve(
        "I cannot log into my account", top_k=3, intent="account_login"
    )
    assert hits
    assert all(hit["intent"] == "account_login" for hit in hits)


def test_invalid_top_k_is_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("help", top_k=0)


def test_query_rows_can_be_excluded_from_corpus():
    full = HistoricalRetriever.from_csv()
    excluded_id = full.records.iloc[0]["interaction_id"]
    filtered = HistoricalRetriever.from_csv(
        exclude_interaction_ids={excluded_id}
    )
    assert excluded_id not in set(filtered.records["interaction_id"])
