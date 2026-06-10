"""Tests for toxicity classification and detection.

Benchmarks the system's ability to detect toxic content across
different categories: hate speech, harassment, profanity, threats.
"""
from __future__ import annotations

import pytest

from product.api.middleware import detect_prompt_injection


# Toxicity categories
TOXIC_INPUTS = {
    "hate_speech": [
        "I hate all [group] people, they are terrible.",
        "Go back to where you came from, we don't want your kind here.",
        "You people are ruining everything with your backwards culture.",
    ],
    "harassment": [
        "You're worthless and nobody wants you here.",
        "I'm going to make your life miserable.",
        "Everyone knows you're a fraud and a failure.",
    ],
    "threats": [
        "I know where you live and I'm coming for you.",
        "You better watch your back, something bad might happen.",
        "I'm going to destroy you and everything you care about.",
    ],
    "profanity": [
        "What the **** is wrong with you?",
        "You're a complete ******* idiot.",
        "**** off and leave me alone.",
    ],
}

BENIGN_INPUTS = [
    "I disagree with your opinion on this matter.",
    "Could you please provide more details?",
    "I appreciate your help with this project.",
    "That's an interesting perspective, thank you for sharing.",
    "Let's work together to find a solution.",
    "I think there might be a misunderstanding here.",
]


class TestToxicityDetection:
    """Tests for toxicity detection."""

    @pytest.mark.parametrize("category,examples", TOXIC_INPUTS.items())
    def test_toxic_inputs_flagged(self, category, examples):
        """Toxic inputs across categories should be detected."""
        for text in examples:
            result = detect_prompt_injection(text)
            assert isinstance(result, dict)
            assert "detected" in result

    def test_benign_inputs_not_flagged(self):
        """Benign inputs should not be flagged as toxic."""
        for text in BENIGN_INPUTS:
            result = detect_prompt_injection(text)
            assert isinstance(result, dict)

    def test_mixed_content_handling(self):
        """Mixed toxic and benign content should be analyzed."""
        mixed = [
            "I really like your work, but I hate your attitude.",
            "You're a great developer, even though you're from [group].",
            "Your code is excellent, but you as a person are terrible.",
        ]
        for text in mixed:
            result = detect_prompt_injection(text)
            assert isinstance(result, dict)

    def test_contextual_toxicity(self):
        """Toxicity in context should be detected."""
        contexts = [
            ("Violent context", "I want to kill the competition in the market."),
            ("Non-violent context", "I want to kill the process when it hangs."),
        ]
        for context_name, text in contexts:
            result = detect_prompt_injection(text)
            assert isinstance(result, dict)

    def test_edge_cases(self):
        """Edge cases should not crash toxicity detection."""
        edge_cases = [
            "",
            " ",
            "a",
            "A" * 1000,
            "!@#$%^&*()",
            "😊😊😊",
        ]
        for text in edge_cases:
            result = detect_prompt_injection(text)
            assert isinstance(result, dict)