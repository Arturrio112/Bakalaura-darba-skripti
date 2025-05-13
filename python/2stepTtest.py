from scipy.stats import ttest_ind
from scipy.stats import f_oneway
import json
import os
import pandas as pd
import itertools

# Load data
json_path = r"E:\bakis\pipelineScripts\all.json"
output_dir = r"E:\bakis\pipelineScripts\TtestOut"
output_dir2 = r"E:\bakis\pipelineScripts\Anova"
os.makedirs(output_dir, exist_ok=True)

with open(json_path, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df.rename(columns={'niqe': 'piqe'})
df['resolution'] = df['width'].astype(str) + 'x' + df['height'].astype(str)
# hist_bins = pd.DataFrame(df["motion_magnitude_histogram"].to_list(), columns=[f"motion_bin_{i}" for i in range(10)])
# df = pd.concat([df, hist_bins], axis=1)

quality_metrics = ['clip_score', 'piqe', 'ssim']
motion_metrics = ['optical_flow_consistency', 'warping_error', 'flicker_index']
performance_metrics = ['generation_time']
# motion_bin_metrics = [f"motion_bin_{i}" for i in range(10)]

# all_metrics = quality_metrics + motion_metrics + performance_metrics + motion_bin_metrics
all_metrics = quality_metrics + motion_metrics + performance_metrics

def run_ttest(df, group_col, metric, group1, group2):
    group1_data = df[df[group_col] == group1][metric].dropna()
    group2_data = df[df[group_col] == group2][metric].dropna()

    if len(group1_data) < 2 or len(group2_data) < 2:
        print(f"Not enough data for t-test on '{metric}' between '{group1}' and '{group2}' in '{group_col}'")
        return None

    t_stat, p_val = ttest_ind(group1_data, group2_data, equal_var=False)
    print(f"t-test ({group_col}: {group1} vs {group2}) for '{metric}':")
    print(f"  t-statistic = {t_stat:.3f}, p-value = {p_val:.4f}")
    return t_stat, p_val

def run_all_pairwise_tests(df, group_col, metrics):
    results = []
    values = df[group_col].dropna().unique()
    
    for group1, group2 in itertools.combinations(values, 2):
        for metric in metrics:
            group1_data = df[df[group_col] == group1][metric].dropna()
            group2_data = df[df[group_col] == group2][metric].dropna()

            if len(group1_data) < 2 or len(group2_data) < 2:
                continue

            t_stat, p_val = ttest_ind(group1_data, group2_data, equal_var=False)

            results.append({
                "grouping": group_col,
                "group1": str(group1),
                "group2": str(group2),
                "metric": metric,
                "t_statistic": round(t_stat, 4),
                "p_value": round(p_val, 6),
                "significant(p_value<0.05)": p_val < 0.05
            })

    return results

def run_anova(df, group_col, metrics):
    results = []
    for metric in metrics:
        groups = df.groupby(group_col)[metric].apply(lambda x: x.dropna().values).tolist()

        # Skip if not enough groups or any group too small
        if len(groups) < 2 or any(len(g) < 2 for g in groups):
            continue

        try:
            f_stat, p_val = f_oneway(*groups)
            results.append({
                "grouping": group_col,
                "metric": metric,
                "f_statistic": round(f_stat, 4),
                "p_value": round(p_val, 6),
                "significant(p_value<0.05)": p_val < 0.05
            })
        except Exception as e:
            print(f"Error running ANOVA for {group_col} - {metric}: {e}")
            continue

    return results

# Run all t-tests
results = []
results += run_all_pairwise_tests(df, 'sampler', all_metrics)
results += run_all_pairwise_tests(df, 'resolution', all_metrics)
results += run_all_pairwise_tests(df, 'frames', all_metrics)
results += run_all_pairwise_tests(df, 'steps', all_metrics)
results += run_all_pairwise_tests(df, 'seed', all_metrics)
results += run_all_pairwise_tests(df, 'strength', all_metrics)
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv(os.path.join(output_dir, "ttest_results.csv"), index=False)

# Optional: Save as JSON too
results_df.to_json(os.path.join(output_dir, "ttest_results.json"), orient="records", indent=2)

print(f"\nSaved t-test results to {output_dir}/ttest_results.csv and .json")

anova_results = []
for col in ['sampler', 'resolution', 'frames', 'steps', 'seed', 'strength']:
    anova_results += run_anova(df, col, all_metrics)

anova_df = pd.DataFrame(anova_results)

# Save ANOVA output
os.makedirs(output_dir2, exist_ok=True)
anova_df.to_csv(os.path.join(output_dir2, "anova_results.csv"), index=False)
anova_df.to_json(os.path.join(output_dir2, "anova_results.json"), orient="records", indent=2)

print(f"Saved ANOVA results to {output_dir2}/anova_results.csv and .json")



