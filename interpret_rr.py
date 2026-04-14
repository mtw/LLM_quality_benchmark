import pandas as pd
import os
from typing import Dict, Any

# --- Configuration ---
RUN_DIR = "benchmark_runs"

def load_data(run_dir: str) -> Dict[str, pd.DataFrame]:
    """Loads the core Round-Robin benchmark datasets."""
    data = {}
    try:
        consensus_path = os.path.join(run_dir, "consensus_ranked_models.csv")
        if not os.path.exists(consensus_path):
            raise FileNotFoundError(f"Consensus file not found at {consensus_path}")

        data['consensus'] = pd.read_csv(consensus_path)

        agreement_path = os.path.join(run_dir, "judge_agreement.csv")
        if not os.path.exists(agreement_path):
            raise FileNotFoundError(f"Agreement file not found at {agreement_path}")

        data['agreement'] = pd.read_csv(agreement_path)

        # Summary data is useful for linking scores back to prompts/tasks
        summary_path = os.path.join(run_dir, "summary.csv")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Summary file not found at {summary_path}")

        data['summary'] = pd.read_csv(summary_path)
        return data

    except Exception as e:
        print(f"\n[ERROR] Could not load required benchmark files. Ensure the directory '{run_dir}' exists and contains 'consensus_ranked_models.csv', 'judge_agreement.csv', and 'summary.csv'.")
        print(f"Details: {e}")
        return None

def interpret_rr_results(data: Dict[str, pd.DataFrame]):
    """Performs a statistically rich interpretation of the loaded RR data."""

    if not data:
        return

    print("\n" + "="*80)
    print("✨ STATISTICALLY RICH ROUND-ROBIN BENCHMARK INTERPRETATION ✨")
    print("="*80 + "\n")

    # --- 1. Overall Consensus Ranking ---
    print("--- 🥇 1. OVERALL MODEL CONSENSUS RANKING 🏆 ---")
    print("Analysis of aggregated scores from all judges.")
    print("-" * 50)

    consensus_df = data['consensus']
    if 'score_mean' in consensus_df.columns:
        # Sort and display the top models based on mean score
        top_models = consensus_df.sort_values(by='score_mean', ascending=False).head(5)
        print("Top 5 Models (Based on Average Score):")
        print(top_models[['model_name', 'score_median', 'score_mean']].to_markdown(index=False))

        # Insight: Standard Deviation suggests consistency. Lower is better/more consistent.
        if 'score_stdev' in consensus_df.columns:
            print("\nConsistency Check:")
            least_consistent = top_models.sort_values(by='score_stdev', ascending=False)
            print("Model with highest score variability (Potential inconsistency):")
            print(f"  -> {least_consistent['model_name'].iloc[0]} (StDev: {least_consistent['score_stdev'].iloc[0]:.3f})")

    else:
        print("[!] WARNING: 'consensus_ranked_models.csv' seems to be missing expected columns (e.g., score_mean). Cannot perform full ranking.")


    # --- 2. Judge Reliability and Bias Check ---
    print("\n" + "="*80)
    print("⚖️ 2. JUDGE RELIABILITY & CONSISTENCY CHECK 🤝")
    print("Analysis of Spearman Rho (Agreement Coefficient): Higher values approach 1.")
    print("-" * 50)

    agreement_df = data['agreement']
    if 'Judge A' in agreement_df.columns and 'Judge B' in agreement_df.columns:
        # For simplicity, we will focus on the overall average correlation if available
        # Assuming the file structure has a summary of pairwise correlations (e.g., one column per pair)
        print("Pairwise Judge Agreement Summary:")
        # Print all columns to see the raw agreement scores
        print(agreement_df.to_markdown(index=False))

        # Interpretation guidance
        print("\n💡 INTERPRETATION GUIDE:")
        print(" - Scores near 1.0: Judges agree strongly on model rankings.")
        print(" - Scores near 0.0: Judge opinions are largely uncorrelated (High Bias/Inconsistency).")
        print(" - Scores below 0.5: Serious warning about judge consistency; results should be treated cautiously.")

    else:
        print("[!] WARNING: 'judge_agreement.csv' structure is unexpected. Cannot analyze judge agreement.")


    # --- 3. Performance Deep Dive (Prompt/Task Specific) ---
    print("\n" + "="*80)
    print("🧠 3. PERFORMANCE DEEP DIVE (SUMMARY ANALYSIS) 🎯")
    print("Identifying model strengths and weaknesses across specific prompts.")
    print("-" * 50)

    summary_df = data['summary']
    if 'task_type' in summary_df.columns:
        # Group by task type and find the average score for each model
        grouped_scores = summary_df.groupby(['task_type', 'model_name'])['score_median'].mean().reset_index()

        print("\nAverage Performance per Task Type (Median Score):")
        print(grouped_scores.pivot(index='task_type', columns='model_name', values='score_median').fillna('-').to_markdown())

        # Example of finding the best model for a specific task type
        best_coder = grouped_scores[grouped_scores['task_type'] == 'coding'].sort_values(by='score_median', ascending=False)
        if not best_coder.empty:
            print("\n✅ Best Model for 'Coding' Tasks:")
            print(f"  -> {best_coder['model_name'].iloc[0]} (Median Score: {best_coder['score_median'].iloc[0]:.2f})")

    else:
        print("[!] WARNING: 'summary.csv' is missing the 'task_type' column or other critical metadata for deep diving.")


if __name__ == "__main__":
    # Check if pandas is installed before proceeding
    try:
        import pandas as pd
    except ImportError:
        print("\n[FATAL ERROR] The 'interpret_rr.py' script requires the 'pandas' library.")
        print("Please run: pip install pandas")
        exit(1)

    loaded_data = load_data(RUN_DIR)
    if loaded_data:
        interpret_rr_results(loaded_data)