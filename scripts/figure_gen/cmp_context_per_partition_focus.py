import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def extract_context_data(base_path):
    """
    Recursively finds and extracts session_usage, the first message,
    and its timestamp from context.json files.
    """
    dir_path = Path(base_path)

    if not dir_path.exists():
        print(f"Error: The directory '{base_path}' does not exist.")
        return []

    all_data = []
    for json_file in dir_path.rglob('context.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                session_usage = data.get("session_usage", {})
                segments = data.get("segments", {})
                message_history = segments.get("message_history", [])

                first_message = None
                timestamp = 0
                if message_history and len(message_history) > 0:
                    first_message = message_history[0]
                    timestamp = first_message.get("timestamp", 0)

                filtered_data = {
                    "session_usage": session_usage,
                    "first_message": first_message,
                    "timestamp": timestamp
                }
                all_data.append(filtered_data)
        except Exception as e:
            print(f"⚠️ An error occurred reading {json_file}: {e}")

    return all_data


def build_dataframe(extracted_data):
    """
    Transforms the extracted list into a pandas DataFrame and maps old keys to new labels.
    """
    # Dictionary mapping original JSON keys to your new requested tags
    rename_map = {
        "master": "master",
        "tools": "tool_definitions",
        "results": "tool_results",
        "index": "data_index",
        "data": "data_fetched",
        "user": "user_msg",
        "assistant": "assistant_msg"
    }

    rows = []
    for context in extracted_data:
        first_msg_text = ""
        first_msg_raw = context.get("first_message")
        if first_msg_raw and "content" in first_msg_raw:
            msg_content_dict = first_msg_raw["content"]
            if isinstance(msg_content_dict, dict):
                first_msg_text = msg_content_dict.get("content", "")
            else:
                first_msg_text = str(msg_content_dict)

        session_usage = context.get("session_usage", {})
        for session_id, stats in session_usage.items():
            row = {
                "session_id": session_id,
                "first_message": first_msg_text,
                "timestamp": context.get("timestamp", 0)
            }
            if isinstance(stats, dict):
                # Map the old keys to the new labels on the fly
                mapped_stats = {rename_map.get(k, k): v for k, v in stats.items()}
                row.update(mapped_stats)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_grouped_metrics_template(df, out_path="avg_utilization_filtered_model.png"):
    """
    Processes the DataFrame and plots the grouped differences using only selected tags.
    """
    if df.empty:
        print("No data available to plot.")
        return

    # 1. Rank timestamps and assign models
    df['occurrence'] = df.groupby('first_message')['timestamp'].rank(method='first').astype(int)

    label_map = {1: 'E4B', 2: '26B'}
    df['model_version'] = df['occurrence'].map(label_map)

    # 2. Define the new metrics to plot (Filtered down to just 2)
    metrics = [
        "data_fetched",
        "assistant_msg"
    ]
    # Ensure only metrics present in the dataframe are used
    metrics = [m for m in metrics if m in df.columns]

    models = ["26B", "E4B"]

    # 3. Compute average metric per category per model
    plot_data = {}

    for model in models:
        plot_data[model] = []
        model_df = df[df['model_version'] == model]

        for metric in metrics:
            avg_metric = model_df[metric].mean()
            plot_data[model].append(avg_metric if pd.notna(avg_metric) else 0.0)

    # 4. Plot grouped bars
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjusted width since there are fewer bars

    x = np.arange(len(metrics))
    bar_width = 0.35

    bars_26b = ax.bar(
        x - bar_width / 2,
        plot_data["26B"],
        width=bar_width,
        label="26B",
        color='#1f77b4'
    )

    bars_e4b = ax.bar(
        x + bar_width / 2,
        plot_data["E4B"],
        width=bar_width,
        label="E4B",
        color='#ff7f0e'
    )

    # Bar labels
    for bars in [bars_26b, bars_e4b]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    # Formatting
    ax.set_title("Average Context Utilization (Data vs. Assistant)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Utilization Metric", fontsize=12)
    ax.set_ylabel("Average Tokens / Count", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)

    # Pad the top of the y-axis so text labels don't get cut off
    if len(plot_data["26B"]) > 0 and len(plot_data["E4B"]) > 0:
        max_val = max(max(plot_data["26B"]), max(plot_data["E4B"]))
        ax.set_ylim(0, max_val * 1.15)

    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Graph successfully generated and saved as '{out_path}'\n")

    # 5. Print raw values mapping to the template's logic
    print("Average Utilization by Metric:")
    for i, metric in enumerate(metrics):
        print(
            f"Metric {metric:17s}: "
            f"26B = {plot_data['26B'][i]:.1f}, "
            f"E4B = {plot_data['E4B'][i]:.1f}"
        )


if __name__ == "__main__":
    target_directory = "../llm/contexts"

    print("Extracting data...")
    raw_data = extract_context_data(target_directory)

    print("Building DataFrame...")
    df = build_dataframe(raw_data)

    print("Plotting grouped metrics...")
    plot_grouped_metrics_template(df)