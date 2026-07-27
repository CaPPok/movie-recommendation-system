"""Interactive walkthrough of every recommendation scenario.

Usage:
    python scripts/demo_cli.py
    python scripts/demo_cli.py --user 1

This script deliberately plays the role of the backend. The engine hands back
only `movie_id`, `score` and a reason code; every title, poster and genre shown
below is joined here from the serving catalogue. That split is the contract, and
seeing it work is the point of the demo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.config import load_config, output_dir  # noqa: E402
from src.recommenders.engine import (  # noqa: E402
    RecommendationEngine,
    RecommendationRequestError,
)

MENU = """
================= DEMO HỆ THỐNG GỢI Ý PHIM =================
  Đang đăng nhập: {session}
------------------------------------------------------------
  1. Đăng nhập bằng user_id  (bỏ trống = chế độ khách)
  2. Tìm phim theo tên  ->  5 phim tương đồng
  3. Nhận gợi ý cho tôi  (hệ thống tự chọn scenario)
  4. Onboarding: chọn thể loại yêu thích
  5. Xem lịch sử xem phim của user
  6. Xem hồ sơ user trong model (ALS factors)
  0. Thoát
------------------------------------------------------------
Chọn: """


class DemoSession:
    """Holds the state a real backend would keep for a signed-in user."""

    def __init__(self) -> None:
        self.user_id: int | None = None
        self.selected_genres: list[str] = []
        self.selected_movie_ids: list[int] = []
        self.recent_interactions: list[dict[str, Any]] = []

    def describe(self) -> str:
        if self.user_id is None:
            return "Khách (chưa đăng nhập)"
        parts = [f"user_id = {self.user_id}"]
        if self.selected_genres:
            parts.append(f"thể loại đã chọn: {', '.join(self.selected_genres)}")
        if self.recent_interactions:
            parts.append(f"{len(self.recent_interactions)} tương tác gần đây")
        return " | ".join(parts)


class Catalogue:
    """Title lookup and metadata join - the backend's half of the contract."""

    def __init__(self, config: dict[str, Any]) -> None:
        serving_dir = output_dir(config, "serving_dir", create=False)
        self.frame = pd.read_parquet(
            serving_dir / "movies_serving.parquet",
            columns=[
                "movie_id",
                "title",
                "release_year",
                "genres",
                "vote_average",
                "vote_count",
            ],
        )
        self.by_id = {
            int(row.movie_id): row for row in self.frame.itertuples(index=False)
        }
        self._lower_titles = self.frame["title"].astype("string").str.lower()
        self.genres = sorted(
            {
                str(genre)
                for values in self.frame["genres"]
                for genre in (
                    values.tolist() if hasattr(values, "tolist") else values
                )
            }
        )

    def search(self, text: str, limit: int = 10) -> pd.DataFrame:
        """Match on title, most-voted first so the famous film comes up top."""
        mask = self._lower_titles.str.contains(text.strip().lower(), regex=False, na=False)
        return (
            self.frame.loc[mask]
            .sort_values("vote_count", ascending=False, kind="mergesort")
            .head(limit)
        )

    def label(self, movie_id: int) -> str:
        row = self.by_id.get(int(movie_id))
        if row is None:
            return f"<không có trong catalog: {movie_id}>"
        year = "" if pd.isna(row.release_year) else f" ({int(row.release_year)})"
        genres = (
            row.genres.tolist() if hasattr(row.genres, "tolist") else list(row.genres)
        )
        genre_text = ", ".join(str(value) for value in genres[:3]) or "—"
        votes = 0 if pd.isna(row.vote_count) else int(row.vote_count)
        score = 0.0 if pd.isna(row.vote_average) else float(row.vote_average)
        return f"{row.title}{year} · {genre_text} · {score:.1f}★ {votes:,} vote"


def show_recommendations(response: Any, catalogue: Catalogue) -> None:
    print()
    print(
        f"  scenario: {response.scenario_applied} | "
        f"loại: {response.recommendation_type} | "
        f"fallback: {response.fallback_level}"
    )
    if not response.recommendations:
        print("  (không có kết quả)")
        return
    for position, item in enumerate(response.recommendations, start=1):
        reason = item["reason_code"]
        context = item.get("reason_context") or {}
        if "source_movie_id" in context:
            source = catalogue.by_id.get(int(context["source_movie_id"]))
            source_title = source.title if source is not None else context["source_movie_id"]
            reason = f"{reason} ← {source_title}"
        elif "genre" in context:
            reason = f"{reason} ({context['genre']})"
        print(f"  {position:2}. {catalogue.label(item['movie_id'])}")
        print(f"      {reason}   score={item['score']}")


def load_user_history(
    config: dict[str, Any], user_id: int, limit: int = 15
) -> pd.DataFrame:
    """Read one user's training interactions using a pushdown filter.

    The split is 230 MB, so the filter matters: reading it whole to keep a few
    dozen rows would make the demo unusable.
    """
    splits_dir = output_dir(config, "splits_dir", create=False)
    frame = pd.read_parquet(
        splits_dir / "interactions_train.parquet",
        filters=[("user_id", "==", int(user_id))],
        columns=["movie_id", "interaction_value", "timestamp"],
    )
    return frame.sort_values("timestamp", ascending=False).head(limit)


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def action_login(session: DemoSession, engine: RecommendationEngine) -> None:
    raw = ask("Nhập user_id (Enter để làm khách): ")
    if not raw:
        session.__init__()
        print("  -> Chế độ khách.")
        return
    try:
        user_id = int(raw)
    except ValueError:
        print("  user_id phải là số nguyên.")
        return
    session.__init__()
    session.user_id = user_id
    known = (
        engine.collaborative is not None and engine.collaborative.knows_user(user_id)
    )
    if known:
        print(f"  -> Đăng nhập user {user_id}. Có trong model ALS: gợi ý cá nhân hoá được.")
    else:
        print(
            f"  -> Đăng nhập user {user_id}. KHÔNG có trong model ALS (cold-start): "
            "hệ thống sẽ tự hạ cấp scenario."
        )


def action_search(session: DemoSession, engine: RecommendationEngine, catalogue: Catalogue) -> None:
    text = ask("Tên phim cần tìm: ")
    if not text:
        return
    matches = catalogue.search(text)
    if matches.empty:
        print("  Không tìm thấy phim nào.")
        return
    print()
    for position, row in enumerate(matches.itertuples(index=False), start=1):
        print(f"  {position:2}. {catalogue.label(int(row.movie_id))}")
    choice = ask("\nChọn số thứ tự để xem phim tương đồng (Enter để bỏ qua): ")
    if not choice.isdigit() or not 1 <= int(choice) <= len(matches):
        return
    movie_id = int(matches.iloc[int(choice) - 1]["movie_id"])
    print(f"\n  Vì bạn đã xem: {catalogue.label(movie_id)}")
    try:
        response = engine.because_you_watched([movie_id], limit=5)
    except RecommendationRequestError as error:
        print(f"  {error}")
        return
    show_recommendations(response, catalogue)
    # Watching a film is an interaction; record it the way a backend would.
    session.recent_interactions.insert(
        0, {"movie_id": movie_id, "event_type": "click", "value": None}
    )


def action_recommend(session: DemoSession, engine: RecommendationEngine, catalogue: Catalogue) -> None:
    if session.user_id is None:
        hint = "guest"
    elif session.recent_interactions or len(session.selected_movie_ids) >= 0:
        known = (
            engine.collaborative is not None
            and engine.collaborative.knows_user(session.user_id)
        )
        hint = "returning_user" if known else "onboarding_user"
    else:
        hint = "onboarding_user"

    request = {
        "user_id": session.user_id,
        "scenario_hint": hint,
        "onboarding_completed": bool(session.selected_genres or session.selected_movie_ids),
        "valid_interaction_count_90d": len(session.recent_interactions) or 30,
        "selected_movie_ids": session.selected_movie_ids,
        "selected_genres": session.selected_genres,
        "recent_interactions": session.recent_interactions,
        "exclude_movie_ids": [],
        "limit": 10,
    }
    print(f"\n  Backend gửi scenario_hint = {hint}")
    try:
        response = engine.recommend(request)
    except RecommendationRequestError as error:
        print(f"  Lỗi request: {error}")
        return
    if response.scenario_applied != hint:
        print(f"  Model hạ cấp: {hint} -> {response.scenario_applied}")
    show_recommendations(response, catalogue)


def action_onboarding(session: DemoSession, engine: RecommendationEngine, catalogue: Catalogue) -> None:
    print("\n  Thể loại có sẵn:")
    for position, genre in enumerate(catalogue.genres, start=1):
        end = "\n" if position % 5 == 0 else ""
        print(f"  {position:2}.{genre:<18}", end=end)
    print()
    raw = ask("\nChọn số thứ tự, cách nhau bởi dấu phẩy (vd 1,4,12): ")
    picked = []
    for token in raw.split(","):
        token = token.strip()
        if token.isdigit() and 1 <= int(token) <= len(catalogue.genres):
            picked.append(catalogue.genres[int(token) - 1])
    if not picked:
        print("  Chưa chọn được thể loại nào.")
        return
    session.selected_genres = picked
    if session.user_id is None:
        session.user_id = 999_000_001
        print(f"  (Tự tạo user_id demo {session.user_id} vì onboarding cần tài khoản)")
    print(f"  -> Đã lưu: {', '.join(picked)}")

    response = engine.recommend(
        {
            "user_id": session.user_id,
            "scenario_hint": "onboarding_user",
            "onboarding_completed": True,
            "valid_interaction_count_90d": 0,
            "selected_movie_ids": session.selected_movie_ids,
            "selected_genres": picked,
            "recent_interactions": [],
            "exclude_movie_ids": [],
            "limit": 10,
        }
    )
    show_recommendations(response, catalogue)


def action_history(session: DemoSession, config: dict[str, Any], catalogue: Catalogue) -> None:
    if session.user_id is None:
        print("  Cần đăng nhập trước.")
        return
    print("  Đang đọc lịch sử...", flush=True)
    history = load_user_history(config, session.user_id)
    if history.empty:
        print(f"  User {session.user_id} không có lịch sử trong tập train.")
        return
    print(f"\n  {len(history)} tương tác gần nhất của user {session.user_id}:")
    for row in history.itertuples(index=False):
        stamp = pd.Timestamp(row.timestamp).date()
        print(f"   {stamp}  {row.interaction_value:.1f}★  {catalogue.label(int(row.movie_id))}")
    session.recent_interactions = [
        {
            "movie_id": int(row.movie_id),
            "event_type": "rating",
            "value": float(row.interaction_value),
        }
        for row in history.itertuples(index=False)
    ]
    print(f"\n  -> Đã nạp {len(session.recent_interactions)} tương tác vào phiên làm việc.")


def action_profile(session: DemoSession, engine: RecommendationEngine) -> None:
    if session.user_id is None:
        print("  Cần đăng nhập trước.")
        return
    if engine.collaborative is None:
        print("  Chưa có model ALS.")
        return
    if not engine.collaborative.knows_user(session.user_id):
        print(
            f"  User {session.user_id} không có trong model ALS. Đây là cold-start: "
            "hệ thống dùng content-based hoặc top-rated thay thế."
        )
        return
    row = engine.collaborative.row_by_user[session.user_id]
    vector = engine.collaborative.user_factors[row]
    print(f"\n  Vector sở thích của user {session.user_id} ({len(vector)} chiều):")
    print("  " + "  ".join(f"{value:+.3f}" for value in vector[:12]) + "  ...")
    print(
        "\n  Đây chính là 'ma trận điểm cho từng user'. Model tự học 64 con số này "
        "\n  từ lịch sử tương tác; không chiều nào được đặt tên trước, và không cần "
        "\n  bất kỳ thông tin nhân khẩu học nào."
    )


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_pipeline.yaml")
    parser.add_argument("--model-config", default="configs/model_serving.yaml")
    parser.add_argument("--user", type=int, default=None)
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    model_config = load_config(arguments.model_config)

    print("Đang nạp model...", flush=True)
    engine = RecommendationEngine(config, model_config)
    catalogue = Catalogue(config)
    print(f"  Artifact: {engine.artifact_versions}")
    print(f"  Catalog: {len(catalogue.frame):,} phim")

    session = DemoSession()
    if arguments.user is not None:
        session.user_id = arguments.user

    actions = {
        "1": lambda: action_login(session, engine),
        "2": lambda: action_search(session, engine, catalogue),
        "3": lambda: action_recommend(session, engine, catalogue),
        "4": lambda: action_onboarding(session, engine, catalogue),
        "5": lambda: action_history(session, config, catalogue),
        "6": lambda: action_profile(session, engine),
    }

    while True:
        choice = ask(MENU.format(session=session.describe()))
        if choice in ("0", ""):
            print("Kết thúc.")
            return 0
        action = actions.get(choice)
        if action is None:
            print("  Lựa chọn không hợp lệ.")
            continue
        action()


if __name__ == "__main__":
    sys.exit(main())
