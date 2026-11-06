#!/usr/bin/env python3
"""Render the numbered figures from checked-in inputs or analytic expressions."""
import argparse
import importlib
from pathlib import Path
import time
import subprocess
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUMERICAL = (4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", default="all", help="all or comma-separated figure numbers")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--format", choices=("pdf", "png", "pgf"), default="pdf")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = range(1, 18) if args.figures == "all" else tuple(map(int, args.figures.split(",")))
    for number in selected:
        start = time.monotonic()
        if number in (1, 2, 3, 13):
            if args.format != "pdf":
                parser.error("Schematic figures support PDF output; use --format pdf")
            root = Path(__file__).resolve().parent
            output_dir = args.output.resolve()
            result = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory=" + str(output_dir),
                    f"figure_{number:02d}.tex",
                ],
                cwd=root / "diagrams",
                capture_output=True,
                text=True,
            )
            (output_dir / f"figure_{number:02d}.build.log").write_text(
                result.stdout + result.stderr
            )
            if result.returncode:
                raise RuntimeError(f"LaTeX failed; see {output_dir}/figure_{number:02d}.build.log")
            print(f"Figure {number:02d}: {output_dir}/figure_{number:02d}.pdf", flush=True)
            continue
        module = importlib.import_module(f"born.figures.figure_{number:02d}")
        figure = module.make_figure()
        output = args.output / f"figure_{number:02d}.{args.format}"
        figure.savefig(output, bbox_inches="tight", dpi=160)
        plt.close("all")
        print(f"Figure {number:02d}: {output} ({time.monotonic()-start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
