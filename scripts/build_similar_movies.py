"""Build the precomputed similar-movies table used by "Because you watched".

Usage:
    python scripts/build_similar_movies.py --config configs/data_pipeline.yaml
    python scripts/build_similar_movies.py --top-n 50 --chunk-size 256

Requires the content artifacts produced by `scripts/build_features.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.config import load_config, output_dir  # noqa: E402
from src.features.similarity import build_similar_movies  # noqa: E402


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_pipeline.yaml")
    parser.add_argument("--model-config", default="configs/model_serving.yaml")
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--w-quality",
        type=float,
        default=None,
        help="Override the quality weight; 0 gives plain cosine ordering.",
    )
    return parser.parse_args(argv)


def _use_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
    arguments = parse_arguments(argv)
    config = load_config(arguments.config)
    model_config = load_config(arguments.model_config)
    if arguments.w_quality is not None:
        model_config.setdefault("similar_movies", {})["w_quality"] = (
            arguments.w_quality
        )

    print("Tính ma trận phim tương tự...", flush=True)
    started = time.perf_counter()
    summary = build_similar_movies(
        config,
        model_config=model_config,
        top_n=arguments.top_n,
        chunk_size=arguments.chunk_size,
    )
    elapsed = time.perf_counter() - started
    summary["build_seconds"] = round(elapsed, 2)

    validation_dir = output_dir(config, "validation_dir")
    lines = [
        "# Similar movies summary",
        "",
        "Bảng láng giềng gần nhất được tính trước cho tính năng "
        '"Because you watched". Điểm là cosine similarity giữa các vector TF-IDF '
        "đã chuẩn hóa L2.",
        "",
        "| Chỉ số | Giá trị |",
        "|---|---:|",
        f"| Phim trong catalog | {summary['movies']:,} |",
        f"| Láng giềng giữ lại mỗi phim | {summary['top_n']} |",
        f"| Tổng số cặp | {summary['rows']:,} |",
        f"| Trung bình láng giềng/phim | {summary['mean_neighbours_per_movie']} |",
        f"| Phim không có láng giềng nào | {summary['movies_without_neighbours']:,} |",
        f"| Điểm nhỏ nhất | {summary['score_min']:.6f} |",
        f"| Điểm lớn nhất | {summary['score_max']:.6f} |",
        f"| Thời gian dựng | {summary['build_seconds']} giây |",
        "",
        "Cặp có điểm bằng 0 bị loại: chúng không mang thông tin tương đồng nào, "
        "và lưu lại sẽ khiến nhiễu trông giống dữ liệu thật.",
        "",
    ]
    (validation_dir / "similar_movies_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    (validation_dir / "similar_movies_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nĐã ghi: {validation_dir / 'similar_movies_summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
