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
    'generation_time': 'ģenerēšanas laiks',
    'optical_flow_consistency': 'optiskās plūsmas konsekvence',
    'warping_error': 'fotometriskās deformācijas kļūda',
    'flicker_index': 'mirgošanas indekss'
}
df.rename(columns=param_display_names, inplace=True)
quality_metrics = [param_display_names.get(m, m) for m in quality_metrics]
print(quality_metrics)
motion_metrics = [param_display_names.get(m, m) for m in motion_metrics]
print(motion_metrics)
performance_metrics = [param_display_names.get(m, m) for m in performance_metrics]
def create_quality_performance_tradeoff(df):
    plt.figure(figsize=(12, 8))

    df['normalizēts_piqe'] = 1 - (df['piqe'] - df['piqe'].min()) / (df['piqe'].max() - df['piqe'].min())
    df['normalizēts_ssim'] = (df['ssim'] - df['ssim'].min()) / (df['ssim'].max() - df['ssim'].min())
    df['normalizēts_clip'] = (df['clip_score'] - df['clip_score'].min()) / (df['clip_score'].max() - df['clip_score'].min())

    df['kvalitātes_vērtējums'] = (df['normalizēts_piqe'] + df['normalizēts_ssim'] + df['normalizēts_clip']) / 3

    scatter = sns.scatterplot(
        data=df, 
        x='ģenerēšanas laiks', 
        y='kvalitātes_vērtējums',
        hue='kadri',  
        size='soļi',
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

    df['izšķirtspēja'] = df['width'].astype(str) + 'x' + df['height'].astype(str)

    plt.figure(figsize=(12, 8))
    scatter2 = sns.scatterplot(
        data=df,
        x='ģenerēšanas laiks',
        y='kvalitātes_vērtējums',
        hue='izšķirtspēja',
        size='kadri',
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
        data=df[df['kvalitātes_vērtējums'] > 0.6], 
        x='ģenerēšanas laiks', 
        y='kvalitātes_vērtējums',
        hue='kadri',  
        size='soļi',
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
        df['izšķirtspēja'] = df['width'].astype(str) + 'x' + df['height'].astype(str)
        if 'izšķirtspēja' not in varying_params:
            varying_params.append('izšķirtspēja')

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
                grouped[f'normalizēts {metric}'] = (grouped[metric] - min_val) / (max_val - min_val)
            else:
                grouped[f'normalizēts {metric}'] = 0.5

            if metric in ['piqe', 'fotometriskās deformācijas kļūda', 'mirgošanas indekss']:
                grouped[f'normalizēts {metric}'] = 1 - grouped[f'normalizēts {metric}']

            impact_data = pd.concat([impact_data, grouped[[param, f'normalizēts {metric}']]])

        impact_data_wide = impact_data.pivot_table(
            index=param,
            values=[col for col in impact_data.columns if 'normalizēts' in col]
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
    steps_values = sorted(df['soļi'].unique())
    
    line_styles = ['-', '--', ':', '-.']
    markers = ['o', 's', '^', 'D', 'x']
    colors = plt.cm.viridis(np.linspace(0, 1, len(steps_values)))
    
    legend_elements = []
    
    for i, sampler in enumerate(samplers):
        sampler_df = df[df['sampler'] == sampler]
        
        for j, steps in enumerate(steps_values):

            step_rows = sampler_df[sampler_df['soļi'] == steps]
            
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

    df['normalizēts_piqe'] = 1 - (df['piqe'] - df['piqe'].min()) / (df['piqe'].max() - df['piqe'].min() + 1e-10)
    df['normalizēts_ssim'] = (df['ssim'] - df['ssim'].min()) / (df['ssim'].max() - df['ssim'].min() + 1e-10)
    df['normalizēts_clip'] = (df['clip_score'] - df['clip_score'].min()) / (df['clip_score'].max() - df['clip_score'].min())

    df['norm_flow'] = (df['optiskās plūsmas konsekvence'] - df['optiskās plūsmas konsekvence'].min()) / (df['optiskās plūsmas konsekvence'].max() - df['optiskās plūsmas konsekvence'].min() + 1e-10)
    df['norm_flicker'] = 1 - (df['mirgošanas indekss'] - df['mirgošanas indekss'].min()) / (df['mirgošanas indekss'].max() - df['mirgošanas indekss'].min() + 1e-10)
    df['norm_warping'] = 1 - (df['fotometriskās deformācijas kļūda'] - df['fotometriskās deformācijas kļūda'].min()) / (df['fotometriskās deformācijas kļūda'].max() - df['fotometriskās deformācijas kļūda'].min() + 1e-10)

    df['kvalitātes_vērtējums'] = (((df['normalizēts_piqe'] + df['normalizēts_ssim'] + df['normalizēts_clip']) / 3)*0.5 + ((df['norm_flow'] + df['norm_flicker'] + df['norm_warping']) / 3)*0.5)

    df['veiktspējas vērtējums'] = 1 - (df['ģenerēšanas laiks'] - df['ģenerēšanas laiks'].min()) / (df['ģenerēšanas laiks'].max() - df['ģenerēšanas laiks'].min() + 1e-10)
    df['kopējais vērtējums'] = df['kvalitātes_vērtējums'] * 0.7 + df['veiktspējas vērtējums'] * 0.3

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
        "kadri",
        "seed",
        "soļi"
    ]
    agg_df = df.groupby(group_cols, as_index=False).agg({
        'piqe': 'mean',
        'ssim': 'mean',
        'optiskās plūsmas konsekvence': 'mean',
        'mirgošanas indekss': 'mean',
        'ģenerēšanas laiks': 'mean',
        'kvalitātes_vērtējums': 'mean',
        'veiktspējas vērtējums': 'mean',
        'kopējais vērtējums': 'max' 
    })

    top_configs = agg_df.sort_values('kopējais vērtējums', ascending=False).head(10)

    plt.figure(figsize=(14, 10))
    y_pos = np.arange(len(top_configs))

    config_labels = [
        f"Sampler: {row['sampler']}, S: {row['soļi']}, H: {row['height']}, W: {row['width']}, F: {row['kadri']}"
        for _, row in top_configs.iterrows()
    ]

    bars = plt.barh(y_pos, top_configs['kopējais vērtējums'], align='center')
    plt.yticks(y_pos, config_labels)
    plt.xlabel('Kopējais rezultāts')
    plt.title('10 labāko parametru kombināciju rezultāti')

    for i, (_, row) in enumerate(top_configs.iterrows()):
        plt.text(
            row['kopējais vērtējums'] + 0.01,
            i,
            f"Qual: {row['kvalitātes_vērtējums']:.2f}, Perf: {row['veiktspējas vērtējums']:.2f}",
            va='center'
        )

    plt.savefig(os.path.join(output_dir, "best_configurations.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_best_configuration_analysis_no_strength(df):

    df['normalizēts_piqe'] = 1 - (df['piqe'] - df['piqe'].min()) / (df['piqe'].max() - df['piqe'].min() + 1e-10)
    df['normalizēts_ssim'] = (df['ssim'] - df['ssim'].min()) / (df['ssim'].max() - df['ssim'].min() + 1e-10)
    df['normalizēts_clip'] = (df['clip_score'] - df['clip_score'].min()) / (df['clip_score'].max() - df['clip_score'].min())

    df['norm_flow'] = (df['optiskās plūsmas konsekvence'] - df['optiskās plūsmas konsekvence'].min()) / (df['optiskās plūsmas konsekvence'].max() - df['optiskās plūsmas konsekvence'].min() + 1e-10)
    df['norm_flicker'] = 1 - (df['mirgošanas indekss'] - df['mirgošanas indekss'].min()) / (df['mirgošanas indekss'].max() - df['mirgošanas indekss'].min() + 1e-10)
    df['norm_warping'] = 1 - (df['fotometriskās deformācijas kļūda'] - df['fotometriskās deformācijas kļūda'].min()) / (df['fotometriskās deformācijas kļūda'].max() - df['fotometriskās deformācijas kļūda'].min() + 1e-10)

    df['kvalitātes_vērtējums'] = (((df['normalizēts_piqe'] + df['normalizēts_ssim'] + df['normalizēts_clip']) / 3)*0.5 + ((df['norm_flow'] + df['norm_flicker'] + df['norm_warping']) / 3)*0.5)

    df['veiktspējas vērtējums'] = 1 - (df['ģenerēšanas laiks'] - df['ģenerēšanas laiks'].min()) / (df['ģenerēšanas laiks'].max() - df['ģenerēšanas laiks'].min() + 1e-10)
    df['kopējais vērtējums'] = df['kvalitātes_vērtējums'] * 0.7 + df['veiktspējas vērtējums'] * 0.3

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
        "kadri",
        "seed",
        "soļi"
    ]
    agg_df = df.groupby(group_cols, as_index=False).agg({
        'piqe': 'mean',
        'ssim': 'mean',
        'optiskās plūsmas konsekvence': 'mean',
        'mirgošanas indekss': 'mean',
        'ģenerēšanas laiks': 'mean',
        'kvalitātes_vērtējums': 'mean',
        'veiktspējas vērtējums': 'mean',
        'kopējais vērtējums': 'max' 
    })

    top_configs = agg_df.sort_values('kopējais vērtējums', ascending=False).head(10)

    plt.figure(figsize=(14, 10))
    y_pos = np.arange(len(top_configs))

    config_labels = [
        f"Sampler: {row['sampler']}, S: {row['soļi']}, H: {row['height']}, W: {row['width']}, F: {row['kadri']}"
        for _, row in top_configs.iterrows()
    ]

    bars = plt.barh(y_pos, top_configs['kopējais vērtējums'], align='center')
    plt.yticks(y_pos, config_labels)
    plt.xlabel('Kopējais rezultāts')
    plt.title('10 labāko parametru kombināciju rezultāti(Ignorējot "strength" parametru)')

    for i, (_, row) in enumerate(top_configs.iterrows()):
        plt.text(
            row['kopējais vērtējums'] + 0.01,
            i,
            f"Qual: {row['kvalitātes_vērtējums']:.2f}, Perf: {row['veiktspējas vērtējums']:.2f}",
            va='center'
        )

    plt.savefig(os.path.join(output_dir, "best_configurations_no_strength.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_motion_analysis_by_frames(df):
    plt.figure(figsize=(14, 10))
    
    frames_values = sorted(df['kadri'].unique())
    samplers = df['sampler'].unique()
    
    line_styles = ['-', '--', ':', '-.']
    markers = ['o', 's', '^', 'D', 'x']
    colors = plt.cm.plasma(np.linspace(0, 1, len(frames_values)))
    
    legend_elements = []

    for i, sampler in enumerate(samplers):
        sampler_df = df[df['sampler'] == sampler]
        
        for j, frames in enumerate(frames_values):
            group = sampler_df[sampler_df['kadri'] == frames]
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
                label=f"{sampler}, Kadri: {frames}"
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
    df['izšķirtspēja'] = df['width'].astype(str) + 'x' + df['height'].astype(str)

    plt.figure(figsize=(14, 10))

    resolutions = sorted(df['izšķirtspēja'].unique())
    samplers = df['sampler'].unique()

    line_styles = ['-', '--', ':', '-.']
    markers = ['o', 's', '^', 'D', 'x']
    colors = plt.cm.cividis(np.linspace(0, 1, len(resolutions)))

    legend_elements = []

    for i, sampler in enumerate(samplers):
        sampler_df = df[df['sampler'] == sampler]

        for j, resolution in enumerate(resolutions):
            group = sampler_df[sampler_df['izšķirtspēja'] == resolution]
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
    df['izšķirtspēja'] = df['width'].astype(str) + 'x' + df['height'].astype(str)

    metrics_by_res = df.groupby(['sampler', 'izšķirtspēja']).agg({
        'clip_score': 'mean',
        'piqe': 'mean',
        'ssim': 'mean'
    }).reset_index()

    def normalize(series):
        return (series - series.min()) / (series.max() - series.min()) if series.max() != series.min() else 0

    metrics_by_res['normalizēts_clip'] = normalize(metrics_by_res['clip_score'])
    metrics_by_res['normalizēts_ssim'] = normalize(metrics_by_res['ssim'])
    metrics_by_res['normalizēts_piqe'] = 1 - normalize(metrics_by_res['piqe'])  

    metrics_by_res['kvalitātes_vērtējums'] = (
        metrics_by_res['normalizēts_clip'] +
        metrics_by_res['normalizēts_ssim'] +
        metrics_by_res['normalizēts_piqe']
    ) / 3

    sns.barplot(
        data=metrics_by_res,
        x='izšķirtspēja',
        y='kvalitātes_vērtējums',
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
        y='optiskās plūsmas konsekvence',
        hue='soļi',
        palette='viridis',
        ax=ax1,
        showfliers=False
    )
    ax1.set_title('(A) Optiskās plūsmas konsekvence pēc "sampler" un "soļi" parametriem', fontsize=16)
    ax1.set_xlabel('Sampler', fontsize=14)
    ax1.set_ylabel('Optiskās plūsmas konsekvence\n(augstāks ir labāks)', fontsize=12, labelpad=10)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='soļi', bbox_to_anchor=(1.01, 1), loc='upper left')

    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(
        data=df,
        x='sampler',
        y='fotometriskās deformācijas kļūda',
        hue='soļi',
        palette='viridis',
        ax=ax2,
        showfliers=False
    )
    ax2.set_title('(B) Fotometriskās deformācijas kļūda pēc "sampler" un "soļi" parametriem', fontsize=16)
    ax2.set_xlabel('Sampler', fontsize=14)
    ax2.set_ylabel('Fotometriskās deformācijas kļūda\n(zemāks ir labāks)', fontsize=14, labelpad=10)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='soļi', bbox_to_anchor=(1.01, 1), loc='upper left')

    ax3 = fig.add_subplot(gs[1, 0])
    sns.boxplot(
        data=df,
        x='sampler',
        y='mirgošanas indekss',
        hue='soļi',
        palette='viridis',
        ax=ax3,
        showfliers=False
    )
    ax3.set_title('(C) Mirgošanas indekss pēc "sampler" un "soļi" parametriem', fontsize=16)
    ax3.set_xlabel('Sampler', fontsize=14)
    ax3.set_ylabel('Mirgošanas indekss (zemāks ir labāks)', fontsize=14, labelpad=10)
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(title='soļi', bbox_to_anchor=(1.01, 1), loc='upper left')

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
    motion_df['norm_flow'] = (motion_df['optiskās plūsmas konsekvence'] - motion_df['optiskās plūsmas konsekvence'].min()) / (motion_df['optiskās plūsmas konsekvence'].max() - motion_df['optiskās plūsmas konsekvence'].min())
    motion_df['norm_warp'] = 1 - (motion_df['fotometriskās deformācijas kļūda'] - motion_df['fotometriskās deformācijas kļūda'].min()) / (motion_df['fotometriskās deformācijas kļūda'].max() - motion_df['fotometriskās deformācijas kļūda'].min())
    motion_df['norm_flicker'] = 1 - (motion_df['mirgošanas indekss'] - motion_df['mirgošanas indekss'].min()) / (motion_df['mirgošanas indekss'].max() - motion_df['mirgošanas indekss'].min())
    motion_df['motion_score'] = (motion_df['norm_flow'] + motion_df['norm_warp'] + motion_df['norm_flicker']) / 3
    motion_by_frames = motion_df.groupby(['kadri', 'sampler'])['motion_score'].mean().reset_index()
    sns.lineplot(
        data=motion_by_frames,
        x='kadri',
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
        x='soļi',
        y='ģenerēšanas laiks',
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
    df['res_frames'] = df['izšķirtspēja'] + ' | ' + df['kadri'].astype(str) + 'F'
    sns.barplot(
        data=df,
        x='res_frames',
        y='ģenerēšanas laiks',
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
    mean_df = df.groupby(['kadri', 'sampler'])['ģenerēšanas laiks'].mean().reset_index()
    sns.lineplot(
        data=mean_df,
        x='kadri',
        y='ģenerēšanas laiks',
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

    df['izšķirtspēja'] = df['width'] * df['height']
    analysis_cols = quality_metrics + motion_metrics + performance_metrics + [
        p for p in ['soļi', 'kadri', 'izšķirtspēja'] if p in df.columns and pd.api.types.is_numeric_dtype(df[p])
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

potential_vars = ['soļi', 'kadri', 'seed', 'strength', 'sampler', 'width', 'height']
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


