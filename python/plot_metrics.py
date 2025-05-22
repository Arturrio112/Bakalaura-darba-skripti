# USED TO MAKE GRAPHS 
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
output_dir = r"E:\bakis\graphs"
os.makedirs(output_dir, exist_ok=True)

with open(json_path, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df.rename(columns={'niqe': 'piqe'}) # CHANGE THE NIQE COLUMN TO PIQE, BECAUSE NIQE WASNT USED FOR METRICS

# Define quality metrics and performance metrics
quality_metrics = ['clip_score', 'piqe', 'ssim']
motion_metrics = ['optical_flow_consistency', 'warping_error', 'flicker_index']
performance_metrics = ['generation_time']
# Needed to display lv instead of eng
param_display_names = {
    'resolution': 'izšķirtspēja',
    'steps': 'soļi',
    'frames': 'kadri',
}
def create_quality_performance_tradeoff(df):
    plt.figure(figsize=(12, 8))

    df['normalized_piqe'] = 1 - (df['piqe'] - df['piqe'].min()) / (df['piqe'].max() - df['piqe'].min())
    df['normalized_ssim'] = (df['ssim'] - df['ssim'].min()) / (df['ssim'].max() - df['ssim'].min())
    df['normalized_clip'] = (df['clip_score'] - df['clip_score'].min()) / (df['clip_score'].max() - df['clip_score'].min())

    df['quality_score'] = (df['normalized_piqe'] + df['normalized_ssim'] + df['normalized_clip']) / 3

    scatter = sns.scatterplot(
        data=df, 
        x='generation_time', 
        y='quality_score',
        hue='frames',  
        size='steps',
        style='sampler',
        palette='viridis',
        sizes=(50, 200),
        alpha=0.7
    )
    
    plt.title('Kvalitātes un veiktspējas kompromisa grafiks')
    plt.xlabel('Ģenerēšanas laiks (sekundēs)')
    plt.ylabel('Kombinētā kvalitātes vērtība (higher is better)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quality_performance_tradeoff.png"), dpi=300)
    plt.close() 

    df['resolution'] = df['width'].astype(str) + 'x' + df['height'].astype(str)

    plt.figure(figsize=(12, 8))
    scatter2 = sns.scatterplot(
        data=df,
        x='generation_time',
        y='quality_score',
        hue='resolution',
        size='frames',
        style='sampler',
        palette='viridis',
        sizes=(50, 200),
        alpha=0.7
    )
    plt.title('Kvalitātes un veiktspējas kompromisa grafiks pēc izšķirtspējas parametriem')
    plt.xlabel('Ģenerēšanas laiks (sekundēs)')
    plt.ylabel('Kombinētā kvalitātes vērtība (augstāks ir labāks)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quality_performance_tradeoff_by_resolution.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 8))
    scatter3 = sns.scatterplot(
        data=df[df['quality_score'] > 0.6], 
        x='generation_time', 
        y='quality_score',
        hue='frames',  
        size='steps',
        style='sampler',
        palette='viridis',
        sizes=(50, 200),
        alpha=0.7
    )
    
    plt.title('Kvalitātes un veiktspējas kompromisa grafiks(Kreisā augšējā stūra palielinājums)')
    plt.xlabel('Ģenerēšanas laiks (sekundēs)')
    plt.ylabel('Kombinētā kvalitātes vērtība (higher is better)')
    plt.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        borderaxespad=0.
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quality_performance_tradeoff_upper_left.png"), dpi=300, bbox_inches='tight')
    plt.close() 

def create_parameter_impact_analysis(df, varying_params):
    df = df.copy()

    if 'height' in df.columns and 'width' in df.columns:
        df['resolution'] = df['width'].astype(str) + 'x' + df['height'].astype(str)
        if 'resolution' not in varying_params:
            varying_params.append('resolution')

    skip_params = {'height', 'width'}

    for param in varying_params:
        if param in skip_params:
            continue

        if param not in df.columns or df[param].nunique() <= 1:
            continue
        param_values = sorted(df[param].unique())
        if len(param_values) <= 1:
            continue

        metrics_to_plot = quality_metrics + motion_metrics
        impact_data = pd.DataFrame()

        for metric in metrics_to_plot:
            grouped = df.groupby(param)[metric].mean().reset_index()
            min_val = grouped[metric].min()
            max_val = grouped[metric].max()

            if max_val > min_val:
                grouped[f'normalized_{metric}'] = (grouped[metric] - min_val) / (max_val - min_val)
            else:
                grouped[f'normalized_{metric}'] = 0.5

            if metric in ['piqe', 'warping_error', 'flicker_index']:
                grouped[f'normalized_{metric}'] = 1 - grouped[f'normalized_{metric}']

            impact_data = pd.concat([impact_data, grouped[[param, f'normalized_{metric}']]])

        impact_data_wide = impact_data.pivot_table(
            index=param,
            values=[col for col in impact_data.columns if 'normalized' in col]
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            impact_data_wide,
            annot=True,
            cmap='RdYlGn',
            cbar_kws={'label': 'Ietekme (augstāka ir labāka)'}
        )
        param_display = param_display_names.get(param, param)
        plt.title(f'"{param_display}" ietekme uz metrikām')
        plt.tight_layout()
        plt.ylabel(param_display)
        plt.savefig(os.path.join(output_dir, f"parameter_impact_{param}.png"), dpi=300)
        plt.close()


def create_motion_analysis2(df):

    plt.figure(figsize=(14, 10))
    

    samplers = df['sampler'].unique()
    steps_values = sorted(df['steps'].unique())
    
    line_styles = ['-', '--', ':', '-.']
    markers = ['o', 's', '^', 'D', 'x']
    colors = plt.cm.viridis(np.linspace(0, 1, len(steps_values)))
    
    legend_elements = []
    
    for i, sampler in enumerate(samplers):
        sampler_df = df[df['sampler'] == sampler]
        
        for j, steps in enumerate(steps_values):

            step_rows = sampler_df[sampler_df['steps'] == steps]
            
            if len(step_rows) == 0:
                continue
                
            if len(step_rows) > 1:
                combined_hist = np.mean([row['motion_magnitude_histogram'] for _, row in step_rows.iterrows()], axis=0)
                hist_data = combined_hist
            else:
                hist_data = step_rows.iloc[0]['motion_magnitude_histogram']

            line = plt.plot(
                range(len(hist_data)), 
                hist_data,
                color=colors[j],
                linestyle=line_styles[i % len(line_styles)],
                marker=markers[i % len(markers)],
                markersize=8,
                alpha=0.8,
                linewidth=2
            )

            legend_label = f"{sampler}, Steps: {steps}"
            legend_elements.append(plt.Line2D([0], [0], 
                                             color=colors[j], 
                                             linestyle=line_styles[i % len(line_styles)],
                                             marker=markers[i % len(markers)],
                                             markersize=8,
                                             linewidth=2,
                                             label=legend_label))
    
    plt.title('Kustības lieluma histogramma pēc “sampler” un “soļi” parametriem', fontsize=16)
    plt.xlabel('Kustības intervāls', fontsize=14)
    plt.ylabel('Frekvence', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.xticks(range(len(hist_data)), range(len(hist_data)))

    plt.legend(handles=legend_elements, 
               loc='center left', 
               bbox_to_anchor=(1.02, 0.5), 
               fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "motion_histogram_by_steps.png"), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

def create_best_configuration_analysis(df):

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

    top_configs = agg_df.sort_values('overall_score', ascending=False).head(10)

    plt.figure(figsize=(14, 10))
    y_pos = np.arange(len(top_configs))

    config_labels = [
        f"Sampler: {row['sampler']}, S: {row['steps']}, H: {row['height']}, W: {row['width']}, F: {row['frames']}"
        for _, row in top_configs.iterrows()
    ]

    bars = plt.barh(y_pos, top_configs['overall_score'], align='center')
    plt.yticks(y_pos, config_labels)
    plt.xlabel('Kopējais rezultāts')
    plt.title('10 labāko parametru kombināciju rezultāti')

    for i, (_, row) in enumerate(top_configs.iterrows()):
        plt.text(
            row['overall_score'] + 0.01,
            i,
            f"Qual: {row['quality_score']:.2f}, Perf: {row['performance_score']:.2f}",
            va='center'
        )

    plt.savefig(os.path.join(output_dir, "best_configurations.png"), dpi=300, bbox_inches='tight')
    plt.close()

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

    plt.figure(figsize=(14, 10))
    y_pos = np.arange(len(top_configs))

    config_labels = [
        f"Sampler: {row['sampler']}, S: {row['steps']}, H: {row['height']}, W: {row['width']}, F: {row['frames']}"
        for _, row in top_configs.iterrows()
    ]

    bars = plt.barh(y_pos, top_configs['overall_score'], align='center')
    plt.yticks(y_pos, config_labels)
    plt.xlabel('Kopējais rezultāts')
    plt.title('10 labāko parametru kombināciju rezultāti(Ignorējot "strength" parametru)')

    for i, (_, row) in enumerate(top_configs.iterrows()):
        plt.text(
            row['overall_score'] + 0.01,
            i,
            f"Qual: {row['quality_score']:.2f}, Perf: {row['performance_score']:.2f}",
            va='center'
        )

    plt.savefig(os.path.join(output_dir, "best_configurations_no_strength.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_motion_analysis_by_frames(df):
    plt.figure(figsize=(14, 10))
    
    frames_values = sorted(df['frames'].unique())
    samplers = df['sampler'].unique()
    
    line_styles = ['-', '--', ':', '-.']
    markers = ['o', 's', '^', 'D', 'x']
    colors = plt.cm.plasma(np.linspace(0, 1, len(frames_values)))
    
    legend_elements = []

    for i, sampler in enumerate(samplers):
        sampler_df = df[df['sampler'] == sampler]
        
        for j, frames in enumerate(frames_values):
            group = sampler_df[sampler_df['frames'] == frames]
            if len(group) == 0:
                continue

            if len(group) > 1:
                hist_data = np.mean([row['motion_magnitude_histogram'] for _, row in group.iterrows()], axis=0)
            else:
                hist_data = group.iloc[0]['motion_magnitude_histogram']
            
            plt.plot(
                range(len(hist_data)),
                hist_data,
                color=colors[j],
                linestyle=line_styles[i % len(line_styles)],
                marker=markers[i % len(markers)],
                markersize=8,
                alpha=0.8,
                linewidth=2
            )
            
            legend_elements.append(plt.Line2D(
                [0], [0],
                color=colors[j],
                linestyle=line_styles[i % len(line_styles)],
                marker=markers[i % len(markers)],
                markersize=8,
                linewidth=2,
                label=f"{sampler}, Frames: {frames}"
            ))

    plt.title('Kustības lieluma histogramma pēc “sampler” un “kadri” parametriem', fontsize=16)
    plt.xlabel('Kustības intervāls', fontsize=14)
    plt.ylabel('Frekvence', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(hist_data)), range(len(hist_data)))
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "motion_histogram_by_frames.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_motion_analysis_by_resolution(df):
    df['resolution'] = df['width'].astype(str) + 'x' + df['height'].astype(str)

    plt.figure(figsize=(14, 10))

    resolutions = sorted(df['resolution'].unique())
    samplers = df['sampler'].unique()

    line_styles = ['-', '--', ':', '-.']
    markers = ['o', 's', '^', 'D', 'x']
    colors = plt.cm.cividis(np.linspace(0, 1, len(resolutions)))

    legend_elements = []

    for i, sampler in enumerate(samplers):
        sampler_df = df[df['sampler'] == sampler]

        for j, resolution in enumerate(resolutions):
            group = sampler_df[sampler_df['resolution'] == resolution]
            if len(group) == 0:
                continue

            if len(group) > 1:
                hist_data = np.mean([row['motion_magnitude_histogram'] for _, row in group.iterrows()], axis=0)
            else:
                hist_data = group.iloc[0]['motion_magnitude_histogram']

            plt.plot(
                range(len(hist_data)),
                hist_data,
                color=colors[j],
                linestyle=line_styles[i % len(line_styles)],
                marker=markers[i % len(markers)],
                markersize=8,
                alpha=0.8,
                linewidth=2
            )

            legend_elements.append(plt.Line2D(
                [0], [0],
                color=colors[j],
                linestyle=line_styles[i % len(line_styles)],
                marker=markers[i % len(markers)],
                markersize=8,
                linewidth=2,
                label=f"{sampler}, Izšķirtspēja: {resolution}"
            ))

    plt.title('Kustības lieluma histogramma pēc “sampler” un izšķirtspējas parametriem', fontsize=16)
    plt.xlabel('Kustības intervāls', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(hist_data)), range(len(hist_data)))
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "motion_histogram_by_resolution.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_quality_metrics_dashboard(df):
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.8, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    sns.barplot(
        data=df,
        x='sampler',
        y='clip_score',
        estimator='mean',
        errorbar='sd',
        hue='sampler',
        palette='viridis',
        ax=ax1
    )
    ax1.set_title('(A) CLIP Score pēc "sampler" parametra', fontsize=14)
    ax1.set_xlabel('Sampler', fontsize=12)
    ax1.set_ylabel('CLIP Score (augstāks ir labāks)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)

    ax2 = fig.add_subplot(gs[0, 1])
    sns.barplot(
        data=df,
        x='sampler',
        y='piqe',
        estimator='mean',
        errorbar='sd',
        hue='sampler',
        palette='viridis',
        ax=ax2
    )
    ax2.set_title('(B) PIQE pēc "sampler" parametra', fontsize=14)
    ax2.set_xlabel('Sampler', fontsize=12)
    ax2.set_ylabel('PIQE (zemāks ir labāks)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)

    ax3 = fig.add_subplot(gs[1, 0])
    sns.barplot(
        data=df,
        x='sampler',
        y='ssim',
        estimator='mean',
        errorbar='sd',
        hue='sampler',
        palette='viridis',
        ax=ax3
    )
    ax3.set_title('(C) SSIM pēc sampler parametra', fontsize=14)
    ax3.set_xlabel('Sampler', fontsize=12)
    ax3.set_ylabel('SSIM (augstāks ir labāks)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)

    ax4 = fig.add_subplot(gs[1, 1])
    df['resolution'] = df['width'].astype(str) + 'x' + df['height'].astype(str)

    metrics_by_res = df.groupby(['sampler', 'resolution']).agg({
        'clip_score': 'mean',
        'piqe': 'mean',
        'ssim': 'mean'
    }).reset_index()

    def normalize(series):
        return (series - series.min()) / (series.max() - series.min()) if series.max() != series.min() else 0

    metrics_by_res['normalized_clip'] = normalize(metrics_by_res['clip_score'])
    metrics_by_res['normalized_ssim'] = normalize(metrics_by_res['ssim'])
    metrics_by_res['normalized_piqe'] = 1 - normalize(metrics_by_res['piqe'])  

    metrics_by_res['quality_score'] = (
        metrics_by_res['normalized_clip'] +
        metrics_by_res['normalized_ssim'] +
        metrics_by_res['normalized_piqe']
    ) / 3

    sns.barplot(
        data=metrics_by_res,
        x='resolution',
        y='quality_score',
        hue='sampler',
        palette='viridis',
        ax=ax4
    )
    ax4.set_title('(D) Kvalitātes vērtējums pēc izšķirtspējas un "sampler" parametra', fontsize=14)
    ax4.set_xlabel('Izšķirtspēja', fontsize=12)
    ax4.set_ylabel('Kvalitātes vērtējums (augstāks ir labāks)', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(title='Sampler', bbox_to_anchor=(1.01, 1), loc='upper left')

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig(os.path.join(output_dir, "quality_metrics_dashboard.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_motion_metrics_dashboard(df):
    fig = plt.figure(figsize=(20, 18)) 
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.2], hspace=0.7, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(
        data=df,
        x='sampler',
        y='optical_flow_consistency',
        hue='steps',
        palette='viridis',
        ax=ax1,
        showfliers=False
    )
    ax1.set_title('(A) Optiskās plūsmas konsekvence pēc "sampler" un "soļi" parametriem', fontsize=16)
    ax1.set_xlabel('Sampler', fontsize=14)
    ax1.set_ylabel('Optiskās plūsmas konsekvence\n(augstāks ir labāks)', fontsize=12, labelpad=10)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='Steps', bbox_to_anchor=(1.01, 1), loc='upper left')

    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(
        data=df,
        x='sampler',
        y='warping_error',
        hue='steps',
        palette='viridis',
        ax=ax2,
        showfliers=False
    )
    ax2.set_title('(B) Fotometriskās deformācijas kļūda pēc "sampler" un "soļi" parametriem', fontsize=16)
    ax2.set_xlabel('Sampler', fontsize=14)
    ax2.set_ylabel('Fotometriskās deformācijas kļūda\n(zemāks ir labāks)', fontsize=14, labelpad=10)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Steps', bbox_to_anchor=(1.01, 1), loc='upper left')

    ax3 = fig.add_subplot(gs[1, 0])
    sns.boxplot(
        data=df,
        x='sampler',
        y='flicker_index',
        hue='steps',
        palette='viridis',
        ax=ax3,
        showfliers=False
    )
    ax3.set_title('(C) Mirgošanas indekss pēc "sampler" un "soļi" parametriem', fontsize=16)
    ax3.set_xlabel('Sampler', fontsize=14)
    ax3.set_ylabel('Mirgošanas indekss (zemāks ir labāks)', fontsize=14, labelpad=10)
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(title='Steps', bbox_to_anchor=(1.01, 1), loc='upper left')

    ax4 = fig.add_subplot(gs[1, 1])
    samplers = df['sampler'].unique()
    colors = plt.cm.plasma(np.linspace(0, 1, len(samplers)))
    for i, sampler in enumerate(samplers):
        sampler_df = df[df['sampler'] == sampler]
        avg_hist = np.mean([row['motion_magnitude_histogram'] for _, row in sampler_df.iterrows()], axis=0)
        ax4.plot(
            range(len(avg_hist)),
            avg_hist,
            label=sampler,
            color=colors[i],
            linewidth=2
        )
    ax4.set_title('(D) Vidēja kustības lieluma histogramma pēc "sampler" parametra', fontsize=16)
    ax4.set_xlabel('Kustības intervāls', fontsize=14)
    ax4.set_ylabel('Frekvence', fontsize=14, labelpad=10)
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    ax5 = fig.add_subplot(gs[2, :]) 
    motion_df = df.copy()
    motion_df['norm_flow'] = (motion_df['optical_flow_consistency'] - motion_df['optical_flow_consistency'].min()) / (motion_df['optical_flow_consistency'].max() - motion_df['optical_flow_consistency'].min())
    motion_df['norm_warp'] = 1 - (motion_df['warping_error'] - motion_df['warping_error'].min()) / (motion_df['warping_error'].max() - motion_df['warping_error'].min())
    motion_df['norm_flicker'] = 1 - (motion_df['flicker_index'] - motion_df['flicker_index'].min()) / (motion_df['flicker_index'].max() - motion_df['flicker_index'].min())
    motion_df['motion_score'] = (motion_df['norm_flow'] + motion_df['norm_warp'] + motion_df['norm_flicker']) / 3
    motion_by_frames = motion_df.groupby(['frames', 'sampler'])['motion_score'].mean().reset_index()
    sns.lineplot(
        data=motion_by_frames,
        x='frames',
        y='motion_score',
        hue='sampler',
        marker='o',
        palette='viridis',
        linewidth=2,
        ax=ax5
    )
    ax5.set_title('(F) Kustības kvalitātes vērtējums pēc "kadri" un "sampler" parametriem', fontsize=16)
    ax5.set_xlabel('Kadru skaits', fontsize=14)
    ax5.set_ylabel('Kustības kvalitātes vērtējums (augstāks ir labāks)', fontsize=14, labelpad=12)
    ax5.grid(True, alpha=0.3)
    ax5.legend(bbox_to_anchor=(1.01, 0.5), loc='center left')

    plt.subplots_adjust(left=0.07, right=0.83, top=0.93, bottom=0.06)
    plt.savefig(os.path.join(output_dir, "motion_metrics_dashboard.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_dashboard(df):
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    sns.lineplot(
        data=df,
        x='steps',
        y='generation_time',
        hue='sampler',
        style='sampler',
        markers=True,
        palette='inferno',
        ax=ax2
    )
    ax2.set_title('(B) Ģenerēšanas laiks pēc "soļi" un "sampler" parametriem', fontsize=14)
    ax2.set_xlabel('Soļi', fontsize=12)
    ax2.set_ylabel('Ģenerēšanas laiks (sekundēs)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    ax3 = fig.add_subplot(gs[1, 0])
    df['res_frames'] = df['resolution'] + ' | ' + df['frames'].astype(str) + 'F'
    sns.barplot(
        data=df,
        x='res_frames',
        y='generation_time',
        hue='sampler',
        palette='inferno',
        ax=ax3
    )
    ax3.set_title('(C) Ģenerēšanas laiks pēc izšķirtspējas, "kadri" un "sampler" parametriem', fontsize=14)
    ax3.set_xlabel('Izšķirtspēja un kadru skaits', fontsize=12)
    ax3.set_ylabel('Ģenerēšanas laiks (s)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.01, 1), loc='upper left')


    ax4 = fig.add_subplot(gs[0, 0])
    mean_df = df.groupby(['frames', 'sampler'])['generation_time'].mean().reset_index()
    sns.lineplot(
        data=mean_df,
        x='frames',
        y='generation_time',
        hue='sampler',
        marker='o',
        palette='inferno',
        ax=ax4
    )
    ax4.set_title('(A) Ģenerēšanas laiks pēc "kadri" un "sampler" parametriem', fontsize=14)
    ax4.set_xlabel('Kadri', fontsize=12)
    ax4.set_ylabel('Vid. Ģenerēšanas laiks (s)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.subplots_adjust(right=0.85, top=0.9)
    plt.savefig(os.path.join(output_dir, "performance_dashboard.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_matrix(df):

    df['resolution'] = df['width'] * df['height']
    analysis_cols = quality_metrics + motion_metrics + performance_metrics + [
        p for p in ['steps', 'frames', 'resolution'] if p in df.columns and pd.api.types.is_numeric_dtype(df[p])
    ]
    
    plt.figure(figsize=(12, 10))
    corr_matrix = df[analysis_cols].corr()
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        center=0,
        fmt='.2f'
    )
    plt.title('Parametru un metriku korelācijas matrica')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300)
    plt.close()

potential_vars = ['steps', 'frames', 'seed', 'strength', 'sampler', 'width', 'height']
varying_params = [param for param in potential_vars if df[param].nunique() > 1]

# Run all analyses
print("Creating quality-performance tradeoff analysis...")
create_quality_performance_tradeoff(df)

print("Creating parameter impact analysis...")
create_parameter_impact_analysis(df, varying_params)

print("Creating motion analysis2...")
create_motion_analysis2(df)

print("Creating motion analysis by frames...")
create_motion_analysis_by_frames(df)

print("Creating motion analysis by resolution...")
create_motion_analysis_by_resolution(df)

print("Creating best configuration analysis...")
create_best_configuration_analysis(df)

print("Creating best configuration analysis without strength parameter...")
create_best_configuration_analysis_no_strength(df)

print("Creating quality dashboard...")
create_quality_metrics_dashboard(df)

print("Creating motion dashboard...")
create_motion_metrics_dashboard(df)

print("Creating performance dashboard...")
create_performance_dashboard(df)

print("Creating correlation matrix...")
create_correlation_matrix(df)

print("Analysis complete! All graphs saved to:", output_dir)


