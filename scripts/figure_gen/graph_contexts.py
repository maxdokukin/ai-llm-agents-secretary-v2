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
    Transforms the extracted list into a pandas DataFrame and processes text/timestamps.
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


def process_and_plot(df):
    """
    Processes the DataFrame to assign E4B/E26B labels and plots the stacked bar chart.
    """
    if df.empty:
        print("No data available to plot.")
        return

    # 1. Truncate prompt to the first 3 words
    df['prompt_3_words'] = df['first_message'].apply(
        lambda x: ' '.join(str(x).split()[:3]) + ('...' if len(str(x).split()) > 3 else '')
    )

    # 2. Sort by timestamp first
    df = df.sort_values('timestamp')

    # 3. Handle duplicates by ranking the timestamp (1 = earlier, 2 = later)
    df['occurrence'] = df.groupby('prompt_3_words')['timestamp'].rank(method='first').astype(int)

    # 4. Map the occurrence to the requested model versions
    # 1 (Earlier) -> E4B, 2 (Later) -> E26B
    label_map = {1: 'E4B', 2: 'E26B'}
    df['model_version'] = df['occurrence'].map(label_map)

    # Create the final Y-axis label using the 3-word prompt and model version
    df['y_label'] = df['prompt_3_words'] + " (" + df['model_version'] + ")"

    # 5. Sort so identical prompts are grouped together, and earlier messages (E4B) are first
    df = df.sort_values(['prompt_3_words', 'timestamp'])

    # 6. Define the utilization columns for the X-axis
    usage_cols = ['master', 'tools', 'results', 'index', 'data', 'user', 'assistant']
    usage_cols = [col for col in usage_cols if col in df.columns]

    # Print a snippet of the DataFrame to verify the labels are correct
    print("\nProcessed DataFrame Sample (Checking Labels):")
    print(df[['prompt_3_words', 'timestamp', 'model_version']].head(4).to_string())
    print("-" * 50)

    # 7. Plotting
    ax = df.plot(
        x='y_label',
        y=usage_cols,
        kind='barh',
        stacked=True,
        figsize=(14, 16),
        colormap='viridis'
    )

    # Ensure E4B appears directly above E26B for each prompt
    ax.invert_yaxis()

    # Formatting
    plt.title("Session Utilization Breakdown per Prompt (E4B vs E26B)", fontsize=16, pad=20)
    plt.xlabel("Utilization Breakdown (Token / Count amounts)", fontsize=12)
    plt.ylabel("Prompt (First 3 Words)", fontsize=12)
    plt.legend(title="Utilization Type", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    output_file = "utilization_graph.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Graph successfully generated and saved as '{output_file}'")


if __name__ == "__main__":
    target_directory = "../llm/contexts"

    print("Extracting data...")
    raw_data = extract_context_data(target_directory)

    print("Building DataFrame...")
    df = build_dataframe(raw_data)

    print("Plotting graph...")
    process_and_plot(df)