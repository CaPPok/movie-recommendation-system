"""Reference implementation of the comment sentiment source, for the backend team.

The model never receives comment text. It receives one number in [-1, 1] and
multiplies it by a weight (see `docs/interaction_events_api.md` section 1.2).
Producing that number is backend's job, and this file shows three ways to do it
so the choice can be made from measurements instead of guesses.

    python scripts/comment_sentiment_provider.py --check-language
    python scripts/comment_sentiment_provider.py --estimate 10000
    python scripts/comment_sentiment_provider.py --text "Phim này rất hay"
    python scripts/comment_sentiment_provider.py --demo

Read `--check-language` first. Amazon Comprehend's DetectSentiment supports a
limited set of languages, and Vietnamese may not be among them. That one call
settles it for your account and region in a second, which is faster than reading
documentation that may be out of date.

The three providers, cheapest first:

* `StarRatingSentiment` — free, no API, most reliable. Use it when the UI collects
  a star rating together with the comment.
* `ComprehendSentiment` — one AWS call per comment. Effectively free at demo
  scale, but only for languages Comprehend supports.
* `TranslateThenComprehendSentiment` — Vietnamese to English, then Comprehend.
  Works when Comprehend alone does not, at two calls and two prices per comment,
  and translation can flatten sarcasm and negation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.config import load_config

# Comprehend bills per unit of 100 characters, with a floor of 3 units (300
# characters) per request, so short comments all cost the same as a 300-character
# one. Verify against the current pricing page before quoting these numbers:
# https://aws.amazon.com/comprehend/pricing/
CHARACTERS_PER_UNIT = 100
MINIMUM_UNITS_PER_REQUEST = 3
PRICE_TIERS = (
    (10_000_000, 0.000_1),
    (50_000_000, 0.000_05),
    (float("inf"), 0.000_025),
)
FREE_TIER_UNITS_PER_MONTH = 50_000  # first 12 months
TRANSLATE_PRICE_PER_MILLION_CHARACTERS = 15.0
TRANSLATE_FREE_CHARACTERS_PER_MONTH = 2_000_000  # first 12 months

# batch_detect_sentiment accepts at most 25 documents per call. Same unit price,
# far fewer round trips, which is what matters when rescoring a backlog.
MAX_BATCH_DOCUMENTS = 25


def sentiment_from_stars(
    stars: float, neutral: float = 3.0, span: float = 2.0
) -> float:
    """Map a 1-5 star rating to sentiment in [-1, 1].

    5 stars -> +1.0, 3 stars -> 0.0, 1 star -> -1.0.

    This is the provider to use if the UI allows it. A star rating is the user
    stating their own opinion; every other option on this page is a machine
    guessing at it.
    """
    return max(-1.0, min(1.0, (float(stars) - neutral) / span))


def billable_units(text: str) -> int:
    """Comprehend units one document costs, including the 3-unit floor."""
    length = len(text or "")
    units = -(-length // CHARACTERS_PER_UNIT)  # ceiling division
    return max(units, MINIMUM_UNITS_PER_REQUEST)


def unit_cost_usd(units: int, units_already_used: int = 0) -> float:
    """Cost of `units` more units this month, walking the pricing tiers."""
    remaining = units
    position = units_already_used
    total = 0.0
    for ceiling, price in PRICE_TIERS:
        if position >= ceiling:
            continue
        available = ceiling - position
        charged = min(remaining, available)
        total += charged * price
        remaining -= charged
        position += charged
        if remaining <= 0:
            break
    return total


@dataclass
class SentimentResult:
    """One classified comment, plus what it cost to classify."""

    sentiment: float | None
    label: str | None = None
    units: int = 0
    characters_translated: int = 0
    error: str | None = None

    def as_event_value(self) -> float | None:
        """The `value` field to put on the `comment` event, or None to omit it.

        Returning None is not a failure to handle: the model drops a comment with
        no sentiment on purpose, because scoring an unclassified opinion as
        positive would recommend more films like the one being complained about.
        """
        return self.sentiment


def score_to_sentiment(scores: dict[str, Any]) -> float:
    """Turn Comprehend's four probabilities into one signed number.

    `Positive - Negative` is used rather than the categorical `Sentiment` label
    because it keeps the magnitude: a 0.95-positive review and a 0.55-positive
    one both come back labelled POSITIVE, and the weight formula has a use for
    the difference. MIXED and NEUTRAL both land near zero, which is the right
    answer for both.
    """
    positive = float(scores.get("Positive", 0.0))
    negative = float(scores.get("Negative", 0.0))
    return max(-1.0, min(1.0, positive - negative))


class ComprehendSentiment:
    """Amazon Comprehend DetectSentiment / BatchDetectSentiment.

    `language_code` must be one Comprehend supports for sentiment. Passing an
    unsupported code fails per call, which is why `check_language_support` exists
    and should be run once before committing to this provider.
    """

    def __init__(
        self,
        aws_config: dict[str, Any],
        language_code: str = "vi",
        client: Any = None,
    ) -> None:
        self.language_code = language_code
        self.units_used = 0
        if client is not None:
            self._client = client
        else:
            from src.aws import s3_sync

            self._client = s3_sync.client(aws_config, "comprehend")

    def check_language_support(self, sample: str = "Phim này rất hay") -> dict[str, Any]:
        """Ask the service directly whether it will classify this language.

        Documentation about supported languages goes stale; the API does not.
        """
        try:
            response = self._client.detect_sentiment(
                Text=sample, LanguageCode=self.language_code
            )
        except Exception as error:  # noqa: BLE001 - the error type is the answer
            name = type(error).__name__
            return {
                "language_code": self.language_code,
                "supported": False,
                "error_type": name,
                "error": str(error),
                "advice": (
                    "Comprehend không phân tích được ngôn ngữ này. Dùng "
                    "StarRatingSentiment, hoặc TranslateThenComprehendSentiment, "
                    "hoặc Amazon Bedrock."
                ),
            }
        return {
            "language_code": self.language_code,
            "supported": True,
            "label": response["Sentiment"],
            "sentiment": round(score_to_sentiment(response["SentimentScore"]), 4),
            "scores": {
                key: round(float(value), 4)
                for key, value in response["SentimentScore"].items()
            },
        }

    def classify(self, text: str) -> SentimentResult:
        units = billable_units(text)
        try:
            response = self._client.detect_sentiment(
                Text=text, LanguageCode=self.language_code
            )
        except Exception as error:  # noqa: BLE001
            # A classification failure must not fail the user's comment write.
            # No sentiment means the comment is dropped by the model, which is
            # the same outcome as not having a sentiment provider at all.
            return SentimentResult(None, units=units, error=str(error))
        self.units_used += units
        return SentimentResult(
            sentiment=score_to_sentiment(response["SentimentScore"]),
            label=response["Sentiment"],
            units=units,
        )

    def classify_batch(self, texts: Sequence[str]) -> list[SentimentResult]:
        """Classify up to 25 documents per call.

        Use this to rescore a backlog of comments before a retraining run; the
        per-unit price is identical but the wall-clock time is not.
        """
        results: list[SentimentResult] = []
        for start in range(0, len(texts), MAX_BATCH_DOCUMENTS):
            chunk = list(texts[start : start + MAX_BATCH_DOCUMENTS])
            units = [billable_units(text) for text in chunk]
            try:
                response = self._client.batch_detect_sentiment(
                    TextList=chunk, LanguageCode=self.language_code
                )
            except Exception as error:  # noqa: BLE001
                results.extend(
                    SentimentResult(None, units=unit, error=str(error))
                    for unit in units
                )
                continue

            ordered: list[SentimentResult] = [
                SentimentResult(None, units=unit) for unit in units
            ]
            for item in response.get("ResultList", []):
                index = int(item["Index"])
                ordered[index] = SentimentResult(
                    sentiment=score_to_sentiment(item["SentimentScore"]),
                    label=item["Sentiment"],
                    units=units[index],
                )
                self.units_used += units[index]
            for item in response.get("ErrorList", []):
                index = int(item["Index"])
                ordered[index] = SentimentResult(
                    None,
                    units=units[index],
                    error=f"{item.get('ErrorCode')}: {item.get('ErrorMessage')}",
                )
            results.extend(ordered)
        return results


class TranslateThenComprehendSentiment:
    """Vietnamese to English via Amazon Translate, then Comprehend.

    The fallback when Comprehend cannot read the source language. Two services,
    two prices, and one real accuracy cost: machine translation tends to flatten
    sarcasm and double negation, which is exactly where sentiment analysis is
    already weakest. Sample the output before trusting it.
    """

    def __init__(
        self,
        aws_config: dict[str, Any],
        source_language: str = "vi",
        target_language: str = "en",
        translate_client: Any = None,
        comprehend: ComprehendSentiment | None = None,
    ) -> None:
        self.source_language = source_language
        self.target_language = target_language
        self.characters_translated = 0
        if translate_client is not None:
            self._translate = translate_client
        else:
            from src.aws import s3_sync

            self._translate = s3_sync.client(aws_config, "translate")
        self._comprehend = comprehend or ComprehendSentiment(
            aws_config, language_code=target_language
        )

    def classify(self, text: str) -> SentimentResult:
        try:
            translated = self._translate.translate_text(
                Text=text,
                SourceLanguageCode=self.source_language,
                TargetLanguageCode=self.target_language,
            )["TranslatedText"]
        except Exception as error:  # noqa: BLE001
            return SentimentResult(
                None, characters_translated=len(text), error=str(error)
            )
        self.characters_translated += len(text)
        result = self._comprehend.classify(translated)
        result.characters_translated = len(text)
        return result


@dataclass
class CostEstimate:
    """What a month of classification would cost at a given volume."""

    comments: int
    average_characters: int
    units: int
    comprehend_usd: float
    comprehend_usd_after_free_tier: float
    translate_usd: float
    translate_usd_after_free_tier: float
    notes: list[str] = field(default_factory=list)


def estimate_cost(comments: int, average_characters: int = 200) -> CostEstimate:
    """Monthly cost for `comments` classifications.

    Note the 3-unit floor: a 50-character comment and a 300-character one cost
    the same, so short comments are where the effective price per character is
    worst.
    """
    units_each = max(
        -(-average_characters // CHARACTERS_PER_UNIT), MINIMUM_UNITS_PER_REQUEST
    )
    units = comments * units_each
    comprehend = unit_cost_usd(units)
    billable_after_free = max(units - FREE_TIER_UNITS_PER_MONTH, 0)
    characters = comments * average_characters
    translate = characters / 1_000_000 * TRANSLATE_PRICE_PER_MILLION_CHARACTERS
    translate_after_free = (
        max(characters - TRANSLATE_FREE_CHARACTERS_PER_MONTH, 0)
        / 1_000_000
        * TRANSLATE_PRICE_PER_MILLION_CHARACTERS
    )

    notes = [
        f"1 unit = {CHARACTERS_PER_UNIT} ký tự, tối thiểu "
        f"{MINIMUM_UNITS_PER_REQUEST} unit/request, nên mỗi comment tính "
        f"{units_each} unit.",
        f"Free tier {FREE_TIER_UNITS_PER_MONTH:,} unit/tháng trong 12 tháng đầu "
        f"= khoảng {FREE_TIER_UNITS_PER_MONTH // units_each:,} comment/tháng.",
    ]
    if units <= FREE_TIER_UNITS_PER_MONTH:
        notes.append("Ở mức này Comprehend nằm trong free tier: 0 USD.")
    return CostEstimate(
        comments=comments,
        average_characters=average_characters,
        units=units,
        comprehend_usd=round(comprehend, 4),
        comprehend_usd_after_free_tier=round(unit_cost_usd(billable_after_free), 4),
        translate_usd=round(translate, 4),
        translate_usd_after_free_tier=round(translate_after_free, 4),
        notes=notes,
    )


def build_comment_event(
    movie_id: int,
    result: SentimentResult,
    timestamp: str,
) -> dict[str, Any]:
    """The event backend writes to DynamoDB and later sends to the model.

    `value` is omitted entirely when there is no sentiment, rather than sent as
    0 or null-with-a-default. Omitting it is what makes the model drop the event
    and report it under `missing_sentiment`.
    """
    event: dict[str, Any] = {
        "movie_id": int(movie_id),
        "event_type": "comment",
        "timestamp": timestamp,
    }
    value = result.as_event_value()
    if value is not None:
        event["value"] = round(value, 4)
    return event


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aws-config", default="configs/aws.yaml")
    parser.add_argument(
        "--check-language",
        action="store_true",
        help="Ask Comprehend whether it supports --language. Run this first.",
    )
    parser.add_argument("--language", default="vi")
    parser.add_argument("--text", default=None, help="Classify one comment.")
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Route --text through Amazon Translate before Comprehend.",
    )
    parser.add_argument(
        "--estimate",
        type=int,
        default=None,
        metavar="COMMENTS",
        help="Estimate the monthly cost for this many comments.",
    )
    parser.add_argument("--average-characters", type=int, default=200)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Show the star-rating provider, which needs no AWS call.",
    )
    return parser.parse_args(argv)


def _use_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
    arguments = parse_arguments(argv)

    if arguments.demo:
        print("StarRatingSentiment — miễn phí, không gọi AWS, chính xác nhất:")
        _print(
            [
                {
                    "stars": stars,
                    "sentiment": round(sentiment_from_stars(stars), 4),
                    "comment_score": round(3 + 12 * sentiment_from_stars(stars), 2),
                }
                for stars in (1.0, 2.0, 3.0, 4.0, 4.5, 5.0)
            ]
        )
        return 0

    if arguments.estimate is not None:
        estimate = estimate_cost(arguments.estimate, arguments.average_characters)
        _print(estimate.__dict__)
        return 0

    aws_config = load_config(REPOSITORY_ROOT / arguments.aws_config)

    try:
        if arguments.check_language:
            provider = ComprehendSentiment(aws_config, arguments.language)
            _print(provider.check_language_support())
            return 0

        if arguments.text:
            if arguments.translate:
                result = TranslateThenComprehendSentiment(
                    aws_config, source_language=arguments.language
                ).classify(arguments.text)
            else:
                result = ComprehendSentiment(
                    aws_config, arguments.language
                ).classify(arguments.text)
            _print(
                {
                    "text": arguments.text,
                    "sentiment": result.sentiment,
                    "label": result.label,
                    "units_billed": result.units,
                    "characters_translated": result.characters_translated,
                    "error": result.error,
                    "comment_score": (
                        round(3 + 12 * result.sentiment, 2)
                        if result.sentiment is not None
                        else None
                    ),
                    "event": build_comment_event(862, result, "2026-07-27T12:00:00Z"),
                }
            )
            return 0
    except Exception as error:  # noqa: BLE001
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2

    print(
        "Chọn một trong: --check-language, --text <chuỗi>, --estimate <số>, --demo",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
