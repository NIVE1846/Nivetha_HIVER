import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from rule_v2 import label_v2  # noqa: E402


def intent(text):
    return label_v2(text)["intent"]


def test_premium_requires_billing_context():
    assert intent("I have Spotify Premium") != "premium_billing"
    assert intent("I was charged twice") == "premium_billing"


def test_download_requires_content_or_offline_context():
    assert intent("I can't download my songs for offline listening") == "download_offline"
    assert intent("I downloaded the Spotify app") != "download_offline"


def test_app_bug_requires_specific_malfunction():
    assert intent("The app crashes when I open it") == "app_bug"
    assert intent("Still not working") is None


def test_specific_intents():
    assert intent("My songs keep skipping") == "playback_issue"
    assert intent("I can't log into my account") == "account_login"
    assert intent("I can't find this album on Spotify") == "content_search"
    assert intent("Can you add this feature?") == "general_inquiry"


def test_metadata_is_auditable():
    result = label_v2("I was charged twice for Spotify Premium")
    assert result["rule_version"] == "rule_v2"
    assert result["is_fallback"] is False
    assert result["matched_rules"]
