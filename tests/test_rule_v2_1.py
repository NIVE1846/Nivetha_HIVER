import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from rule_v2_1 import label_v2_1  # noqa: E402


def intent(text):
    return label_v2_1(text)["intent"]


def test_playback_paraphrases():
    assert intent("My songs keep cutting out") == "playback_issue"
    assert intent("I love this song") is None


def test_app_state_paraphrases():
    assert intent("The Spotify app is frozen on a blank screen") == "app_bug"
    assert intent("My phone screen is blue") is None


def test_account_access_paraphrases():
    assert intent("I can't get into my profile") == "account_login"
    assert intent("My account is for Spotify") is None


def test_content_availability_paraphrases():
    assert intent("I can't listen to this album") == "content_search"
    assert intent("This song is my favorite") is None


def test_offline_saved_content_paraphrases():
    assert intent("My downloaded songs are missing") == "download_offline"
    assert intent("I downloaded the Spotify app") is None


def test_billing_context_is_required():
    assert intent("Am I eligible for the family plan?") == "premium_billing"
    assert intent("I have Spotify Premium") is None


def test_general_information_context_is_required():
    assert intent("Is there a way to use Spotify in my car?") == "general_inquiry"
    assert intent("Spotify is great") is None


def test_v2_metadata_is_preserved_and_versioned():
    result = label_v2_1("I can't get into my profile")
    assert result["rule_version"] == "rule_v2_1"
    assert result["is_fallback"] is False
    assert result["matched_rules"]
