import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "benchmarks" / "courtvision_nba_detection_v1"


def draw_box(draw, box, scale, color, width=3):
    draw.rectangle(tuple(value * scale for value in box), outline=color, width=width)


def render_pages(benchmark_dir, analysis, output_dir, page_size=20, columns=4):
    data_root = benchmark_dir / "data" / analysis.get("split", "test")
    output_dir.mkdir(parents=True, exist_ok=True)
    for previous_page in output_dir.glob("errors_*.jpg"):
        previous_page.unlink()
    cases = analysis["cases"]
    rows = math.ceil(min(page_size, max(1, len(cases))) / columns)
    tile_width, image_height, caption_height = 320, 320, 62
    pages = []
    font = ImageFont.load_default()
    for page_index, start in enumerate(range(0, len(cases), page_size), 1):
        page_cases = cases[start : start + page_size]
        rows = math.ceil(len(page_cases) / columns)
        sheet = Image.new("RGB", (columns * tile_width, rows * (image_height + caption_height)), "white")
        for offset, case in enumerate(page_cases):
            source = Image.open(data_root / case["file_name"]).convert("RGB")
            source.thumbnail((tile_width, image_height))
            scale = source.width / 640.0
            draw = ImageDraw.Draw(source)
            for truth in case["truths"]:
                draw_box(draw, truth["bbox"], scale, "#00ff66")
            for prediction in case["predictions_at_threshold"]:
                draw_box(draw, prediction["bbox"], scale, "#ff3344", 2)
                x, y = prediction["bbox"][:2]
                draw.text((x * scale, max(0, y * scale - 12)), f"{prediction['confidence']:.2f}", fill="#ff3344", font=font)
            x_offset = (offset % columns) * tile_width
            y_offset = (offset // columns) * (image_height + caption_height)
            sheet.paste(source, (x_offset, y_offset))
            miss_reasons = ", ".join(item["reason"] for item in case["missed_truths"]) or "none"
            caption = (
                f"id={case['image_id']} misses={len(case['missed_truths'])} "
                f"fps={len(case['false_positives'])}\n{miss_reasons}\n"
                "green=truth red=prediction"
            )
            ImageDraw.Draw(sheet).text(
                (x_offset + 4, y_offset + image_height + 4),
                caption,
                fill="black",
                font=font,
            )
        output_path = output_dir / f"errors_{page_index:02d}.jpg"
        sheet.save(output_path, quality=90)
        pages.append(output_path)
    return pages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render NBA detector failure review sheets.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    benchmark_dir = args.benchmark_dir.resolve()
    analysis_path = args.analysis or benchmark_dir / "error_analysis_test.json"
    output_dir = args.output_dir or benchmark_dir / "review"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    for path in render_pages(benchmark_dir, analysis, output_dir):
        print(path)
