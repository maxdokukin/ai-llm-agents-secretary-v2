import json
import pandas as pd
import matplotlib.pyplot as plt
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
    Transforms the extracted list into a pandas DataFrame.
    """
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
                row.update(stats)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_differences_template(df, out_path="utilization_difference_by_model.png"):
    """
    Processes the DataFrame and plots the differences following the requested template.
    """
    if df.empty:
        print("No data available to plot.")
        return

    # 1. Rank timestamps and assign models
    df['occurrence'] = df.groupby('first_message')['timestamp'].rank(method='first').astype(int)

    # Map to match your template's naming convention
    label_map = {1: 'E4B', 2: '26B'}
    df['model_version'] = df['occurrence'].map(label_map)

    # 2. Compute a single utilization metric per row (sum of all usage stats)
    usage_cols = ['master', 'tools', 'results', 'index', 'data', 'user', 'assistant']
    available_cols = [col for col in usage_cols if col in df.columns]
    df['total_utilization'] = df[available_cols].sum(axis=1)

    # 3. Setup Template Variables
    models = ["26B", "E4B"]
    avg_utilization = {}

    # Calculate Average Total Utilization per model
    for model in models:
        model_data = df[df['model_version'] == model]
        avg_val = model_data['total_utilization'].mean()
        avg_utilization[model] = avg_val if pd.notna(avg_val) else 0

    # Percent change for 26B over E4B
    if avg_utilization["E4B"] > 0:
        percent_change_26b_over_e4b = ((avg_utilization["26B"] - avg_utilization["E4B"]) / avg_utilization["E4B"]) * 100
    else:
        percent_change_26b_over_e4b = 0.0

    # 4. Plot using the requested template formatting
    fig, ax = plt.subplots(figsize=(7, 5))

    x = range(len(models))
    values = [avg_utilization[m] for m in models]

    bars = ax.bar(x, values, width=0.55)

    # Labels on bars
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    # Percent annotation
    if percent_change_26b_over_e4b >= 0:
        annotation = f"26B is {percent_change_26b_over_e4b:.1f}% higher than E4B"
    else:
        annotation = f"26B is {abs(percent_change_26b_over_e4b):.1f}% lower than E4B"

    # Handle y-axis max safely to position the annotation
    y_max = max(values) if max(values) > 0 else 1

    ax.text(
        0.5,
        y_max * 1.12,
        annotation,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

    # Formatting
    ax.set_title("Average Total Context Utilization by Model", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Context Utilization")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylim(0, y_max * 1.3)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Graph successfully generated and saved as '{out_path}'\n")
    plt.show()

    # Terminal Output mimicking the template
    print("Average Context Utilization (Total tokens/counts):")
    for model in models:
        print(f"{model}: {avg_utilization[model]:.2f}")

    print(f"\nPercent change for 26B over E4B: {percent_change_26b_over_e4b:.1f}%")


if __name__ == "__main__":
    target_directory = "../llm/contexts"

    raw_data = extract_context_data(target_directory)
    df = build_dataframe(raw_data)

    plot_differences_template(df)