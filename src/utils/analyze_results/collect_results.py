import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional
import natsort
import pandas as pd


#================ CHANGE IF NEEDED ================
DEFAULT_TIME_BOUND_SECONDS = 3599.0
GRAPH_CLASSES = ["BA", "ER", "RGG", "SBM", "WS"]

def should_include_graph_name(graph_name: str) -> bool:
    """Return True when the graph name matches the expected synthetic pattern."""
    match = re.search(r"setting(\d+)", graph_name)
    if not match:
        return False
    setting_x = int(match.group(1))
    return (setting_x % 5 < 3) and (setting_x % 25 < 20)
#==================================================


def read_result_json(json_path: str) -> Optional[dict[str, Any]]:
    """Read a result JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except Exception as error:
        print(f"Error reading {json_path}: {error}")
        return None


def is_result_successful(
    data: Optional[dict[str, Any]],
    time_bound_seconds: float = DEFAULT_TIME_BOUND_SECONDS,
) -> bool:
    """Return True when the result is present, not failed, and within the time bound."""
    if data is None:
        return False

    try:
        info = data.get("config", {}).get("info", "")
        if ("Terminate" not in info) and ("Fail" in info):
            return False

        runtime = data.get("time")
        if runtime is not None and float(runtime) >= time_bound_seconds:
            return False
        return True
    except Exception:
        return True


def _graph_directories(class_dir: Path) -> list[Path]:
    return [
        graph_dir
        for graph_dir in natsort.natsorted(class_dir.iterdir())
        if graph_dir.is_dir() and should_include_graph_name(graph_dir.name)
    ]


def _load_json_files(graph_folder: Path) -> dict[str, Optional[dict[str, Any]]]:
    return {
        json_file.stem: read_result_json(str(json_file))
        for json_file in graph_folder.glob("*.json")
    }


def _result_row(
    graph_name: str,
    method: str,
    data: Optional[dict[str, Any]],
    time_bound_seconds: float,
) -> dict[str, Any]:
    if is_result_successful(data, time_bound_seconds=time_bound_seconds):
        return {
            "graph_name": graph_name,
            "method": method,
            "density": data.get("density") if isinstance(data, dict) else None,
            "time": data.get("time") if isinstance(data, dict) else None,
            "status": "success",
        }
    return {
        "graph_name": graph_name,
        "method": method,
        "density": None,
        "time": None,
        "status": "failed",
    }


def collect_class_results(
    graph_class: str,
    output_base: str = "output/synthetic",
    time_bound_seconds: float = DEFAULT_TIME_BOUND_SECONDS,
) -> pd.DataFrame:
    class_dir = Path(output_base) / graph_class
    if not class_dir.exists():
        print(f"Directory {class_dir} does not exist")
        return pd.DataFrame()

    rows = []
    for graph_folder in _graph_directories(class_dir):
        for method, data in _load_json_files(graph_folder).items():
            rows.append(_result_row(graph_folder.name, method, data, time_bound_seconds))

    return pd.DataFrame(rows)


def get_available_methods(graph_class: str, output_base: str = "output/synthetic") -> set[str]:
    class_dir = Path(output_base) / graph_class
    if not class_dir.exists():
        return set()

    methods: set[str] = set()
    for graph_folder in _graph_directories(class_dir):
        methods.update(json_file.stem for json_file in graph_folder.glob("*.json"))
    return methods


def create_complete_results(
    graph_class: str,
    output_base: str = "output/synthetic",
    time_bound_seconds: float = DEFAULT_TIME_BOUND_SECONDS,
) -> pd.DataFrame:
    class_dir = Path(output_base) / graph_class
    if not class_dir.exists():
        print(f"Directory {class_dir} does not exist")
        return pd.DataFrame()

    all_methods = natsort.natsorted(get_available_methods(graph_class, output_base))
    rows = []
    for graph_folder in _graph_directories(class_dir):
        existing_results = _load_json_files(graph_folder)
        for method in all_methods:
            rows.append(
                _result_row(
                    graph_folder.name,
                    method,
                    existing_results.get(method),
                    time_bound_seconds,
                )
            )

    return pd.DataFrame(rows)


def _save_pivot_table(df: pd.DataFrame, value_column: str, output_file: Path) -> None:
    if df.empty or not {"graph_name", "method", value_column}.issubset(df.columns):
        return

    pivot = df.pivot_table(
        values=value_column,
        index="graph_name",
        columns="method",
        aggfunc="first",
    )
    pivot = pivot.reindex(natsort.natsorted(pivot.index))
    pivot.to_csv(output_file)
    print(f"Saved {value_column} pivot table to {output_file}")


def save_results(df: pd.DataFrame, graph_class: str, output_dir: str = "output/synthetic") -> None:
    output_path = Path(output_dir) / graph_class
    output_path.mkdir(parents=True, exist_ok=True)

    csv_file = output_path / f"{graph_class}_results.csv"
    json_file = output_path / f"{graph_class}_results.json"
    df.to_csv(csv_file, index=False)
    df.to_json(json_file, orient="records", indent=2)
    print(f"Saved CSV to {csv_file}")
    print(f"Saved JSON to {json_file}")

    _save_pivot_table(df, "density", output_path / f"{graph_class}_density_table.csv")
    _save_pivot_table(df, "time", output_path / f"{graph_class}_time_table.csv")


def collect() -> None:
    parser = argparse.ArgumentParser(description="Collect results from synthetic graph experiments")
    parser.add_argument(
        "graph_class",
        nargs="?",
        type=str,
        choices=GRAPH_CLASSES + ["all"],
        default="all",
        help='Graph class to process (or "all" for all classes)',
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="output/synthetic",
        help="Base directory containing output results",
    )
    parser.add_argument(
        "--save-base",
        type=str,
        default="results/synthetic",
        help="Base directory containing saved results",
    )
    parser.add_argument(
        "--complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include all method-graph combinations (mark missing as failed)",
    )
    parser.add_argument(
        "--time-bound",
        type=float,
        default=DEFAULT_TIME_BOUND_SECONDS,
        help="Runtime bound in seconds; results with runtime >= this value are marked as failed",
    )

    args = parser.parse_args()
    classes = GRAPH_CLASSES if args.graph_class == "all" else [args.graph_class]
    collector = create_complete_results if args.complete else collect_class_results

    for graph_class in classes:
        print(f"\n{'=' * 60}")
        print(f"Processing class: {graph_class}")
        print(f"{'=' * 60}")

        df = collector(
            graph_class,
            args.output_base,
            time_bound_seconds=args.time_bound,
        )

        if df.empty:
            print(f"No results found for {graph_class}")
            continue

        print(f"\nFound {len(df)} results")
        print(f"Graphs: {df['graph_name'].nunique()}")
        print(f"Methods: {df['method'].nunique()}")

        if "status" in df.columns:
            print("\nStatus summary:")
            print(df["status"].value_counts())

        save_results(df, graph_class, args.save_base)


if __name__ == "__main__":
    collect()
