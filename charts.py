"""Plot category metrics from a saved evaluation summary."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary")
    parser.add_argument("output")
    args = parser.parse_args()
    with Path(args.summary).open(encoding="utf-8") as file:
        rows = json.load(file)
    if not rows:
        raise ValueError("Summary is empty")
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    from matplotlib.figure import Figure

    labels = [f"{row['model']} | {row['category']}" for row in rows]
    fig = Figure(figsize=(11, max(4, len(rows) * 0.4)))
    axis = fig.subplots()
    for index, row in enumerate(rows):
        if row["rate"] is None:
            axis.text(0.01, index, "No scored trials", va="center")
        else:
            errors = [
                [max(0, row["rate"] - row["wilson_95_low"])],
                [max(0, row["wilson_95_high"] - row["rate"])],
            ]
            axis.barh(index, row["rate"], xerr=errors, capsize=3, color="tab:blue")
        if row.get("inconclusive_trials", 0):
            labels[index] += f" [scored {row['scored_trials']}/{row['trials']}]"
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Expected-behavior failure rate (95% Wilson interval; scored trials only)")
    axis.invert_yaxis()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)


if __name__ == "__main__":
    main()
