# HELPER FUNCTIONS FOR PRECISE AVERAGES, DEVIATIONS AND OTHER THINGS
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
# Load data
json_path = r"E:\bakis\pipelineScripts\all.json"
def normalize(series):
    return (series - series.min()) / (series.max() - series.min()) if series.max() != series.min() else 0

with open(json_path, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df.rename(columns={'niqe': 'piqe'}) # CHANGE THE NIQE COLUMN TO PIQE, BECAUSE NIQE WASNT USED FOR METRICS
df['resolution'] = df['width'].astype(str) + 'x' + df['height'].astype(str)
df['normalized_clip'] = normalize(df['clip_score'])
df['normalized_ssim'] = normalize(df['ssim'])
df['normalized_piqe'] = 1 - normalize(df['piqe'])  

df['quality_score'] = (df['normalized_clip'] + df['normalized_ssim'] + df['normalized_piqe']) / 3
df['norm_flow'] = (df['optical_flow_consistency'] - df['optical_flow_consistency'].min()) / (df['optical_flow_consistency'].max() - df['optical_flow_consistency'].min())
df['norm_warp'] = 1 - (df['warping_error'] - df['warping_error'].min()) / (df['warping_error'].max() - df['warping_error'].min())
df['norm_flicker'] = 1 - (df['flicker_index'] - df['flicker_index'].min()) / (df['flicker_index'].max() - df['flicker_index'].min())
df['motion_score'] = (df['norm_flow'] + df['norm_warp'] + df['norm_flicker']) / 3
# Define quality metrics and performance metrics
quality_metrics = ['clip_score', 'piqe', 'ssim']
motion_metrics = ['optical_flow_consistency', 'warping_error', 'flicker_index']
performance_metrics = ['generation_time']


def print_motion_linear_expression(df):
    x_all = df['steps'].values.reshape(-1, 1)
    y_all = df['generation_time'].values

    if len(x_all) > 1:
        model_all = LinearRegression().fit(x_all, y_all)
        slope_all = model_all.coef_[0]
        intercept_all = model_all.intercept_
        print(f'Overall: y = {slope_all:.1f}x + {intercept_all:.1f}')
    samplers = df['sampler'].unique()

    for sampler in samplers:
        sub_df = df[df['sampler'] == sampler]
        x = sub_df['steps'].values.reshape(-1, 1)
        y = sub_df['generation_time'].values

        if len(x) > 1:
            model = LinearRegression().fit(x, y)
            slope = model.coef_[0]
            intercept = model.intercept_
            print(f'{sampler}: y = {slope:.1f}x + {intercept:.1f}')

def print_columns_avg_std_for_metric(df, category_col, metric_col):
    
    grouped = df.groupby(category_col)[metric_col]
    print(f'{metric_col} by {category_col}:')
    for category, values in grouped:
        mean = values.mean()
        std = values.std()
        lower = mean - std
        upper = mean + std

        print(
            f' - {category}: '
            f'Mean = {mean:.4f}, '
            f'Std Dev = {std:.4f}, '
            f'Std Min = {lower:.4f} '
            f'Std Max = {upper:.4f} '
        )

def print_grouped_metric_stats(df, group_by_cols, metric_col):
    grouped = df.groupby(group_by_cols)[metric_col]
    print(f'{metric_col} by {" + ".join(group_by_cols)}:')
    
    for group_keys, values in grouped:
        group_str = ', '.join(f'{col}={val}' for col, val in zip(group_by_cols, group_keys))
        mean = values.mean()
        std = values.std()
        lower = mean - std
        upper = mean + std
        median = values.median()
        print(
            f' - {group_str}: '
            f'Mean = {mean:.4f}, '
            f'Median = {median:.4f}, '
            f'Std Dev = {std:.4f}, '
            f'Std Min = {lower:.4f}, '
            f'Std Max = {upper:.4f}'
        )

def print_box_plot_statistics(df, group_by_cols, metric):
    grouped = df.groupby(group_by_cols)[metric]

    print(f'{metric} by {" + ".join(group_by_cols)}:')

    for group_keys, values in grouped:
        group_str = ', '.join(f'{col}={val}' for col, val in zip(group_by_cols, group_keys if isinstance(group_keys, tuple) else [group_keys]))
        
        q1 = values.quantile(0.25)
        median = values.quantile(0.5)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        min_val = values.min()
        max_val = values.max()

        print(
            f' - {group_str}: '
            f'Min = {min_val:.4f}, '
            f'Q1 = {q1:.4f}, '
            f'Median = {median:.4f}, '
            f'Q3 = {q3:.4f}, '
            f'Max = {max_val:.4f}, '
            f'IQR = {iqr:.4f}'
        )

print("---------------------------------------------------------------")
print("Getting avg, min, max and std for quality metrics dashboard...")
print_columns_avg_std_for_metric(df, 'sampler', 'clip_score')
print_columns_avg_std_for_metric(df, 'sampler', 'piqe')
print_columns_avg_std_for_metric(df, 'sampler', 'ssim')
print_grouped_metric_stats(df, ['resolution','sampler'], 'quality_score')
print("---------------------------------------------------------------")
print("Getting avg, min, max and std for motion metrics dashboard...")
print_box_plot_statistics(df, ['sampler','steps'], 'optical_flow_consistency')
print_box_plot_statistics(df, ['sampler','steps'], 'warping_error')
print_box_plot_statistics(df, ['sampler','steps'], 'flicker_index')
print("---------------------------------------------------------------")
print("Getting linear expression for samplers over generation time...")
print_motion_linear_expression(df)
print_grouped_metric_stats(df, ['sampler','frames'], 'motion_score')
print_grouped_metric_stats(df, ['sampler','steps'], 'generation_time')