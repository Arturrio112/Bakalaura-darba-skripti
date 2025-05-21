# USED TO MAKE STATISTICAL ANALYSIS TESTS FOR TTEST AND ONE WAY ANOVA
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
import json
import os
import pandas as pd
import itertools

# Load data
json_path = r"E:\bakis\pipelineScripts\all.json"
output_dir2 = r"E:\bakis\pipelineScripts\Anova"
os.makedirs(output_dir2, exist_ok=True)

with open(json_path, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df.rename(columns={'niqe': 'piqe'})
# Making resolution column from height and width columns
df['resolution'] = df['width'].astype(str) + 'x' + df['height'].astype(str)

# Setting metrics columns
quality_metrics = ['clip_score', 'piqe', 'ssim']
motion_metrics = ['optical_flow_consistency', 'warping_error', 'flicker_index']
performance_metrics = ['generation_time']
all_metrics = quality_metrics + motion_metrics + performance_metrics

# Function that runs anova test for each variable columns for each metric
def run_anova(df, group_col, metrics):
    results = []
    for metric in metrics:
        groups = df.groupby(group_col)[metric].apply(lambda x: x.values).tolist()
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

# Run all anova tests for parametrs
anova_results = []
for col in ['sampler', 'resolution', 'frames', 'steps', 'seed', 'strength']:
    anova_results += run_anova(df, col, all_metrics)

anova_df = pd.DataFrame(anova_results)

# Save ANOVA output
os.makedirs(output_dir2, exist_ok=True)
anova_df.to_json(os.path.join(output_dir2, "anova_results.json"), orient="records", indent=2)

print(f"Saved ANOVA results to {output_dir2}/anova_results.json")



