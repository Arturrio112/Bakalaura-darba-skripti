# GETS THE BEST 20 VIDEOS with and without strength parameter
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
# Load data
json_path = r"E:\bakis\pipelineScripts\all.json"
output_dir = r"E:\bakis\BEST20"
video_root = r"E:\bakis\stable-diffusion-webui\outputs\img2img-images\text2video"
os.makedirs(output_dir, exist_ok=True)
merge_keys = ['model', 'sampler', 'steps', 'frames', 'width', 'height', 'seed', 'strength', 'niqe', 'ssim', 'optical_flow_consistency', 'flicker_index', 'generation_time']
merge_keys2 = ['model', 'sampler', 'steps', 'frames', 'width', 'height', 'seed', 'strength']
def normalize(series):
    return (series - series.min()) / (series.max() - series.min()) if series.max() != series.min() else 0

def create_best_configuration_analysis(df, num):

    df['normalized_piqe'] = 1 - (df['piqe'] - df['piqe'].min()) / (df['piqe'].max() - df['piqe'].min() + 1e-10)
    df['normalized_ssim'] = (df['ssim'] - df['ssim'].min()) / (df['ssim'].max() - df['ssim'].min() + 1e-10)
    df['normalized_clip'] = (df['clip_score'] - df['clip_score'].min()) / (df['clip_score'].max() - df['clip_score'].min())

    df['norm_flow'] = (df['optical_flow_consistency'] - df['optical_flow_consistency'].min()) / (df['optical_flow_consistency'].max() - df['optical_flow_consistency'].min() + 1e-10)
    df['norm_flicker'] = 1 - (df['flicker_index'] - df['flicker_index'].min()) / (df['flicker_index'].max() - df['flicker_index'].min() + 1e-10)
    df['norm_warping'] = 1 - (df['warping_error'] - df['warping_error'].min()) / (df['warping_error'].max() - df['warping_error'].min() + 1e-10)

    df['quality_score'] = (((df['normalized_piqe'] + df['normalized_ssim'] + df['normalized_clip']) / 3)*0.5 + ((df['norm_flow'] + df['norm_flicker'] + df['norm_warping']) / 3)*0.5)

    df['performance_score'] = 1 - (df['generation_time'] - df['generation_time'].min()) / (df['generation_time'].max() - df['generation_time'].min() + 1e-10)
    df['overall_score'] = df['quality_score'] * 0.7 + df['performance_score'] * 0.3

    group_cols = [
        "prompt",
        "n_prompt",
        "model",
        "cfg_scale",
        "eta",
        "batch_count",
        "do_vid2vid",
        "vid2vid_startFrame",
        "inpainting_frames",
        "inpainting_weights",
        "fps",
        "add_soundtrack",
        "soundtrack_path",
        "width",
        "height",
        "sampler",
        "strength",
        "frames",
        "seed",
        "steps"
    ]
    agg_df = df.groupby(group_cols, as_index=False).agg({
        'piqe': 'mean',
        'ssim': 'mean',
        'optical_flow_consistency': 'mean',
        'flicker_index': 'mean',
        'generation_time': 'mean',
        'quality_score': 'mean',
        'performance_score': 'mean',
        'overall_score': 'max' 
    })

    top_configs = agg_df.sort_values('overall_score', ascending=False).head(num)
    return top_configs

def create_best_configuration_analysis_no_strength(df):

    df['normalized_piqe'] = 1 - (df['piqe'] - df['piqe'].min()) / (df['piqe'].max() - df['piqe'].min() + 1e-10)
    df['normalized_ssim'] = (df['ssim'] - df['ssim'].min()) / (df['ssim'].max() - df['ssim'].min() + 1e-10)
    df['normalized_clip'] = (df['clip_score'] - df['clip_score'].min()) / (df['clip_score'].max() - df['clip_score'].min())

    df['norm_flow'] = (df['optical_flow_consistency'] - df['optical_flow_consistency'].min()) / (df['optical_flow_consistency'].max() - df['optical_flow_consistency'].min() + 1e-10)
    df['norm_flicker'] = 1 - (df['flicker_index'] - df['flicker_index'].min()) / (df['flicker_index'].max() - df['flicker_index'].min() + 1e-10)
    df['norm_warping'] = 1 - (df['warping_error'] - df['warping_error'].min()) / (df['warping_error'].max() - df['warping_error'].min() + 1e-10)

    df['quality_score'] = (((df['normalized_piqe'] + df['normalized_ssim'] + df['normalized_clip']) / 3)*0.5 + ((df['norm_flow'] + df['norm_flicker'] + df['norm_warping']) / 3)*0.5)

    df['performance_score'] = 1 - (df['generation_time'] - df['generation_time'].min()) / (df['generation_time'].max() - df['generation_time'].min() + 1e-10)
    df['overall_score'] = df['quality_score'] * 0.7 + df['performance_score'] * 0.3

    group_cols = [
        "prompt",
        "n_prompt",
        "model",
        "cfg_scale",
        "eta",
        "batch_count",
        "do_vid2vid",
        "vid2vid_startFrame",
        "inpainting_frames",
        "inpainting_weights",
        "fps",
        "add_soundtrack",
        "soundtrack_path",
        "width",
        "height",
        "sampler",
        # "strength",
        "frames",
        "seed",
        "steps"
    ]
    agg_df = df.groupby(group_cols, as_index=False).agg({
        'piqe': 'mean',
        'ssim': 'mean',
        'optical_flow_consistency': 'mean',
        'flicker_index': 'mean',
        'generation_time': 'mean',
        'quality_score': 'mean',
        'performance_score': 'mean',
        'overall_score': 'max' 
    })

    top_configs = agg_df.sort_values('overall_score', ascending=False).head(10)
    top_configs['strength']=0.5 # Added to get single video instead of 40
    return top_configs

with open(json_path, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df.rename(columns={'niqe': 'piqe'})

best20 = create_best_configuration_analysis(df,10)
best20 = best20.rename(columns={'piqe': 'niqe'})

all_folders = [os.path.join(video_root, f) for f in os.listdir(video_root) if os.path.isdir(os.path.join(video_root, f))]

folder_metrics = []
for folder in all_folders:
    metrics_path = os.path.join(folder, "metrics.json")
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data['__folder__'] = folder
        folder_metrics.append(data)
    except Exception as e:
        print(f"Failed to read {metrics_path}: {e}")

metrics_df = pd.DataFrame(folder_metrics)

merged = best20.merge(metrics_df, on=merge_keys, how='left')

print("\nMatched folders for best 10:")
for i, row in merged.iterrows():
    folder = row.get('__folder__')
    rank = i + 1
    print(f"{rank:02d}. {folder}")

    folder_name = os.path.basename(folder)
    dest_folder = os.path.join(output_dir, f"Rating_{rank}" )

    try:
        shutil.copytree(folder, dest_folder)

        mp4_files = [f for f in os.listdir(dest_folder) if f.lower().endswith('.mp4')]
        if mp4_files:
            old_mp4_path = os.path.join(dest_folder, mp4_files[0])
            new_mp4_path = os.path.join(dest_folder, f"Rating {rank}.mp4")
            os.rename(old_mp4_path, new_mp4_path)
        else:
            print(f"No .mp4 file found in {dest_folder} to rename.")
    except Exception as e:
            print(f"Failed to copy or rename folder {folder}: {e}")

best20_no_strength = create_best_configuration_analysis_no_strength(df)
best20_no_strength = best20_no_strength.rename(columns={'piqe': 'niqe'})
all_folders = [os.path.join(video_root, f) for f in os.listdir(video_root) if os.path.isdir(os.path.join(video_root, f))]

folder_metrics = []
for folder in all_folders:
    metrics_path = os.path.join(folder, "metrics.json")
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data['__folder__'] = folder
        folder_metrics.append(data)
    except Exception as e:
        print(f"Failed to read {metrics_path}: {e}")

metrics_df = pd.DataFrame(folder_metrics)

merged = best20_no_strength.merge(metrics_df, on=merge_keys2, how='left')

print("\nMatched folders for best 10(no_strength and unique videos):")
for i, row in merged.iterrows():
    folder = row.get('__folder__')
    rank = i + 1
    print(f"{rank:02d}. {folder}")
    folder_name = os.path.basename(folder)
    dest_folder = os.path.join(output_dir, f"Rating_{rank}(no_strength and unique)" )

    try:
        shutil.copytree(folder, dest_folder)

        mp4_files = [f for f in os.listdir(dest_folder) if f.lower().endswith('.mp4')]
        if mp4_files:
            old_mp4_path = os.path.join(dest_folder, mp4_files[0])
            new_mp4_path = os.path.join(dest_folder, f"Rating {rank}(no_strength and unique).mp4")
            os.rename(old_mp4_path, new_mp4_path)
        else:
            print(f"No .mp4 file found in {dest_folder} to rename.")
    except Exception as e:
            print(f"Failed to copy or rename folder {folder}: {e}")