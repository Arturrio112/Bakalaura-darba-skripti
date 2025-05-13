import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec

# Load data
json_path = r"E:\bakis\pipelineScripts\all.json"
output_dir = r"E:\bakis\graphs"
os.makedirs(output_dir, exist_ok=True)

with open(json_path, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df.rename(columns={'niqe': 'piqe'})
# Define quality metrics and performance metrics
quality_metrics = ['clip_score', 'piqe', 'ssim']
motion_metrics = ['optical_flow_consistency', 'warping_error', 'flicker_index']
performance_metrics = ['generation_time']

# 1. Quality vs Performance Tradeoff Analysis
def create_quality_performance_tradeoff(df):
    plt.figure(figsize=(12, 8))
    
    # Create a composite quality score (normalize and combine metrics)
    df['normalized_piqe'] = 1 - (df['piqe'] - df['piqe'].min()) / (df['piqe'].max() - df['piqe'].min())
    df['normalized_ssim'] = (df['ssim'] - df['ssim'].min()) / (df['ssim'].max() - df['ssim'].min())
    df['normalized_clip'] = (df['clip_score'] - df['clip_score'].min()) / (df['clip_score'].max() - df['clip_score'].min())
    
    # Composite score (higher is better)
    df['quality_score'] = (df['normalized_piqe'] + df['normalized_ssim'] + df['normalized_clip']) / 3
    
    # Plot quality vs generation time
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
    
    # # Add annotations for points
    # for i, row in df.iterrows():
    #     plt.annotate(
    #         f"Steps: {row['steps']}, Strength: {row['strength']}", 
    #         (row['generation_time'], row['quality_score']),
    #         xytext=(5, 5),
    #         textcoords='offset points',
    #         fontsize=8
    #     )
    
    plt.title('Quality vs Performance Tradeoff by Steps and Sampler')
    plt.xlabel('Generation Time (seconds)')
    plt.ylabel('Composite Quality Score (higher is better)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quality_performance_tradeoff.png"), dpi=300)
    plt.close()

    # --- Second Graph: Resolution-based Analysis ---
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
    plt.title('Quality vs Performance Tradeoff by Resolution')
    plt.xlabel('Generation Time (seconds)')
    plt.ylabel('Composite Quality Score (higher is better)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quality_performance_tradeoff_by_resolution.png"), dpi=300)
    plt.close()

# 2. Parameter Impact Analysis
def create_parameter_impact_analysis(df, varying_params):
    df = df.copy()

    # Combine width and height into a 'resolution' column like "256x256"
    if 'height' in df.columns and 'width' in df.columns:
        df['resolution'] = df['width'].astype(str) + 'x' + df['height'].astype(str)
        if 'resolution' not in varying_params:
            varying_params.append('resolution')

    # Skip analyzing width and height separately
    skip_params = {'height', 'width'}

    for param in varying_params:
        if param in skip_params:
            continue

        # if not pd.api.types.is_numeric_dtype(df[param]) and param != 'resolution':
        #     continue
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
            cbar_kws={'label': 'Normalized Impact (higher is better)'}
        )
        plt.title(f'Impact of {param} on Various Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"parameter_impact_{param}.png"), dpi=300)
        plt.close()

# 3. Motion Analysis
def create_motion_analysis(df):
    # Plot motion magnitude histograms
    plt.figure(figsize=(14, 10))  # Increased figure size to accommodate legend
    
    # Extract motion histograms and organize by sampler
    samplers = df['sampler'].unique()
    
    for i, sampler in enumerate(samplers):
        sampler_df = df[df['sampler'] == sampler]
        
        for j, row in sampler_df.iterrows():
            hist_data = row['motion_magnitude_histogram']
            plt.plot(
                range(len(hist_data)), 
                hist_data, 
                label=f"{sampler} (Steps: {row['steps']}, Strength: {row['strength']})",
                alpha=0.7,
                marker='o' if i % 2 == 0 else 's',
                linestyle='-' if j % 2 == 0 else '--'
            )
    
    plt.title('Motion Magnitude Histogram Comparison')
    plt.xlabel('Motion Magnitude Bucket')
    plt.ylabel('Frequency')
    
    # Move legend outside the plot to avoid tight layout issues
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save with adjusted bbox_inches to ensure everything fits
    plt.savefig(os.path.join(output_dir, "motion_histogram_comparison.png"), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

def create_motion_analysis2(df):
    # Plot motion magnitude histograms based on steps
    plt.figure(figsize=(14, 10))
    
    # Get unique samplers and steps
    samplers = df['sampler'].unique()
    steps_values = sorted(df['steps'].unique())
    
    # Use different line styles for different samplers and colors for different steps
    line_styles = ['-', '--', ':', '-.']
    markers = ['o', 's', '^', 'D', 'x']
    colors = plt.cm.viridis(np.linspace(0, 1, len(steps_values)))
    
    # Create a legend handle list
    legend_elements = []
    
    # Plot histograms grouped by sampler and steps
    for i, sampler in enumerate(samplers):
        sampler_df = df[df['sampler'] == sampler]
        
        for j, steps in enumerate(steps_values):
            # Filter by steps
            step_rows = sampler_df[sampler_df['steps'] == steps]
            
            if len(step_rows) == 0:
                continue
                
            # Use first row if multiple with same configuration
            if len(step_rows) > 1:
                # If there are multiple with same configuration, average them
                combined_hist = np.mean([row['motion_magnitude_histogram'] for _, row in step_rows.iterrows()], axis=0)
                hist_data = combined_hist
            else:
                hist_data = step_rows.iloc[0]['motion_magnitude_histogram']
            
            # Plot with consistent color for steps and line style for sampler
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
            
            # Add to legend elements (but only once per combination)
            legend_label = f"{sampler}, Steps: {steps}"
            legend_elements.append(plt.Line2D([0], [0], 
                                             color=colors[j], 
                                             linestyle=line_styles[i % len(line_styles)],
                                             marker=markers[i % len(markers)],
                                             markersize=8,
                                             linewidth=2,
                                             label=legend_label))
    
    plt.title('Motion Magnitude Histogram Comparison by Steps', fontsize=16)
    plt.xlabel('Motion Magnitude Bucket', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add x-tick labels (0 to 9 for the histogram buckets)
    plt.xticks(range(len(hist_data)), range(len(hist_data)))
    
    # Create custom legend with sampler and steps information
    plt.legend(handles=legend_elements, 
               loc='center left', 
               bbox_to_anchor=(1.02, 0.5), 
               fontsize=12)
    
    # Add annotation explaining the histogram
    # plt.figtext(0.5, 0.01, 
    #             "Lower motion magnitude buckets represent smaller movements, higher buckets represent larger movements.", 
    #             ha='center', 
    #             fontsize=10, 
    #             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.5))
    
    # Save with adjusted bbox_inches to ensure everything fits
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "motion_histogram_by_steps.png"), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

    # # Also create a separate violin plot showing distribution of motion magnitudes by steps
    # plt.figure(figsize=(12, 8))
    
    # # Prepare data for the violin plot
    # violin_data = []
    # violin_labels = []
    # violin_steps = []
    # violin_samplers = []
    
    # for i, sampler in enumerate(samplers):
    #     sampler_df = df[df['sampler'] == sampler]
        
    #     for j, steps in enumerate(steps_values):
    #         step_rows = sampler_df[sampler_df['steps'] == steps]
            
    #         if len(step_rows) == 0:
    #             continue
                
    #         # Extract histogram data
    #         for _, row in step_rows.iterrows():
    #             hist = row['motion_magnitude_histogram']
                
    #             # Convert histogram to actual data points for violin plot
    #             # Each histogram bin represents a motion magnitude, repeat it based on frequency
    #             for bin_idx, frequency in enumerate(hist):
    #                 # Scale frequency to get reasonable number of points (multiply by 100)
    #                 points = [bin_idx] * int(frequency * 100)
    #                 violin_data.extend(points)
    #                 violin_labels.extend([f"{sampler}, Steps: {steps}"] * len(points))
    #                 violin_steps.extend([steps] * len(points))
    #                 violin_samplers.extend([sampler] * len(points))
    
    # # Create DataFrame for the violin plot
    # violin_df = pd.DataFrame({
    #     'Motion Magnitude': violin_data,
    #     'Configuration': violin_labels,
    #     'Steps': violin_steps,
    #     'Sampler': violin_samplers
    # })
    
    # # Create the violin plot
    # if not violin_df.empty:
    #     sns.violinplot(
    #         data=violin_df,
    #         x='Steps',
    #         y='Motion Magnitude',
    #         hue='Sampler',
    #         palette='viridis',
    #         split=True if len(samplers) == 2 else False,
    #         inner='quartile',
    #         cut=0
    #     )
        
    #     plt.title('Distribution of Motion Magnitudes by Steps and Sampler', fontsize=16)
    #     plt.xlabel('Steps', fontsize=14)
    #     plt.ylabel('Motion Magnitude', fontsize=14)
    #     plt.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(output_dir, "motion_magnitude_violin_plot.png"), dpi=300)
    #     plt.close()
# 4. Comparative Analysis Dashboard
def create_comparative_dashboard(df):
    # Create a comprehensive dashboard comparing samplers and parameter settings
    fig = plt.figure(figsize=(18, 14))  # Increased size to accommodate all elements
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)  # Added spacing between subplots
    
    # 1. Quality metrics by sampler and steps
    ax1 = fig.add_subplot(gs[0, 0])
    sampler_steps_quality = df.groupby(['sampler', 'steps'])[quality_metrics].mean().reset_index()
    
    # Melt the data for easier plotting
    melted = pd.melt(
        sampler_steps_quality, 
        id_vars=['sampler', 'steps'], 
        value_vars=quality_metrics,
        var_name='Metric', 
        value_name='Value'
    )
    
    sns.barplot(data=melted, x='sampler', y='Value', hue='Metric', ax=ax1)
    ax1.set_title('Quality Metrics by Sampler and Steps')
    ax1.set_xlabel('Sampler')
    ax1.set_ylabel('Metric Value')
    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # 2. Motion metrics by sampler and steps
    ax2 = fig.add_subplot(gs[0, 1])
    sampler_steps_motion = df.groupby(['sampler', 'steps'])[motion_metrics].mean().reset_index()
    
    # Melt the data for easier plotting
    melted = pd.melt(
        sampler_steps_motion, 
        id_vars=['sampler', 'steps'], 
        value_vars=motion_metrics,
        var_name='Metric', 
        value_name='Value'
    )
    
    sns.barplot(data=melted, x='sampler', y='Value', hue='Metric', ax=ax2)
    ax2.set_title('Motion Metrics by Sampler and Steps')
    ax2.set_xlabel('Sampler')
    ax2.set_ylabel('Metric Value')
    ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # 3. Generation time by steps and sampler
    ax3 = fig.add_subplot(gs[1, 0])
    if 'steps' in df.columns and df['steps'].nunique() > 1:
        sns.lineplot(
            data=df, 
            x='steps', 
            y='generation_time', 
            hue='sampler',
            marker='o',
            ax=ax3
        )
        ax3.set_title('Generation Time vs Steps by Sampler')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Generation Time (seconds)')
        ax3.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # 4. SSIM vs Flicker Index - Quality vs Motion Stability tradeoff
    ax4 = fig.add_subplot(gs[1, 1])
    scatter = sns.scatterplot(
        data=df, 
        x='flicker_index', 
        y='ssim',
        hue='sampler',
        size='steps',
        style='model',
        ax=ax4
    )
    ax4.set_title('Quality (SSIM) vs Motion Stability (Flicker Index)')
    ax4.set_xlabel('Flicker Index (lower is better)')
    ax4.set_ylabel('SSIM (higher is better)')
    ax4.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # 5. Optical Flow Consistency vs Steps - Motion quality by steps
    ax5 = fig.add_subplot(gs[2, 0])
    if 'steps' in df.columns and df['steps'].nunique() > 1:
        sns.lineplot(
            data=df, 
            x='steps', 
            y='optical_flow_consistency', 
            hue='sampler',
            marker='o',
            ax=ax5
        )
        ax5.set_title('Optical Flow Consistency vs Steps')
        ax5.set_xlabel('Steps')
        ax5.set_ylabel('Optical Flow Consistency')
        ax5.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # 6. piqe vs Strength - Image quality by strength parameter
    ax6 = fig.add_subplot(gs[2, 1])
    if 'frames' in df.columns and df['frames'].nunique() > 1:
        sns.lineplot(
            data=df, 
            x='frames', 
            y='generation_time', 
            hue='sampler',
            marker='o',
            ax=ax6
        )
        ax6.set_title('Generation Time vs Frames by Sampler')
        ax6.set_xlabel('Frames')
        ax6.set_ylabel('Generation Time (seconds)')
        ax6.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # Save with adjusted bbox_inches to ensure everything fits
    plt.savefig(os.path.join(output_dir, "comparative_dashboard.png"), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
#7 dashboards 4th graph but seperate for each sampler
def plot_quality_vs_motion_stability_by_sampler(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import numpy as np
    from matplotlib.gridspec import GridSpec
    
    # Count unique samplers to determine layout
    unique_samplers = df['sampler'].unique()
    num_samplers = len(unique_samplers)
    
    # Create the figure with proper dimensions
    fig = plt.figure(figsize=(18, 14))  # Increased size to accommodate all elements
    gs = GridSpec(2, 2, figure=fig)  
    
    # # 1. SSIM by Model and Sampler (Quality metric)
    # ax1 = fig.add_subplot(gs[0, 0])
    # sns.boxplot(
    #     data=df, 
    #     x='model', 
    #     y='ssim', 
    #     hue='sampler', 
    #     ax=ax1
    # )
    # ax1.set_title('SSIM by Model and Sampler')
    # ax1.set_xlabel('Model')
    # ax1.set_ylabel('SSIM (higher is better)')
    # ax1.tick_params(axis='x', rotation=45)
    # ax1.legend(title='Sampler', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # # 2. Flicker Index by Model and Sampler (Motion stability)
    # ax2 = fig.add_subplot(gs[0, 1])
    # sns.boxplot(
    #     data=df, 
    #     x='model', 
    #     y='flicker_index', 
    #     hue='sampler',
    #     ax=ax2
    # )
    # ax2.set_title('Flicker Index by Model and Sampler')
    # ax2.set_xlabel('Model')
    # ax2.set_ylabel('Flicker Index (lower is better)')
    # ax2.tick_params(axis='x', rotation=45)
    # ax2.legend(title='Sampler', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # # 3. Steps Impact on Quality (SSIM)
    # ax3 = fig.add_subplot(gs[1, 0])
    # sns.lineplot(
    #     data=df, 
    #     x='steps', 
    #     y='ssim', 
    #     hue='model', 
    #     style='sampler',
    #     markers=True, 
    #     ax=ax3
    # )
    # ax3.set_title('Steps Impact on Quality (SSIM)')
    # ax3.set_xlabel('Steps')
    # ax3.set_ylabel('SSIM (higher is better)')
    # ax3.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # 4. SSIM vs Flicker Index - Quality vs Motion Stability tradeoff
    ax4 = fig.add_subplot(gs[0, 0])  # Top-left corner of the grid
    sns.scatterplot(
        data=df, 
        x='flicker_index', 
        y='ssim',
        hue='sampler',
        size='steps',
        ax=ax4
    )
    ax4.set_title('Quality (SSIM) vs Motion Stability (Flicker Index)')
    ax4.set_xlabel('Flicker Index (lower is better)')
    ax4.set_ylabel('SSIM (higher is better)')
    ax4.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # 5. Sampler-specific Quality vs Motion Stability plots
    for i, sampler in enumerate(unique_samplers):
        if i == 0:
            row, col = 0, 1  # First sampler goes to (0, 1)
        elif i == 1:
            row, col = 1, 0  # Second sampler goes to (1, 0)
        elif i == 2:
            row, col = 1, 1
        
        subset = df[df['sampler'] == sampler]
        ax = fig.add_subplot(gs[row, col])  # Plot in grid
        sns.scatterplot(
            data=subset,
            x='flicker_index',
            y='ssim',
            size='steps',
            ax=ax
        )
        ax.set_title(f'Quality vs Motion Stability - {sampler}')
        ax.set_xlabel('Flicker Index (lower is better)')
        ax.set_ylabel('SSIM (higher is better)')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    filename = "ssim_vs_flicker_per_sampler_dashboard.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
# 5. Best configuration analysis
def create_best_configuration_analysis(df):
    # Define scoring function for overall quality
    df['normalized_piqe'] = 1 - (df['piqe'] - df['piqe'].min()) / (df['piqe'].max() - df['piqe'].min() + 1e-10)
    df['normalized_ssim'] = (df['ssim'] - df['ssim'].min()) / (df['ssim'].max() - df['ssim'].min() + 1e-10)
    df['normalized_clip'] = (df['clip_score'] - df['clip_score'].min()) / (df['clip_score'].max() - df['clip_score'].min())

    df['norm_flow'] = (df['optical_flow_consistency'] - df['optical_flow_consistency'].min()) / (df['optical_flow_consistency'].max() - df['optical_flow_consistency'].min() + 1e-10)
    df['norm_flicker'] = 1 - (df['flicker_index'] - df['flicker_index'].min()) / (df['flicker_index'].max() - df['flicker_index'].min() + 1e-10)
    df['norm_warping'] = 1 - (df['warping_error'] - df['warping_error'].min()) / (df['warping_error'].max() - df['warping_error'].min() + 1e-10)

    # Quality and performance scores
    df['quality_score'] = (((df['normalized_piqe'] + df['normalized_ssim'] + df['normalized_clip']) / 3)*0.5 + ((df['norm_flow'] + df['norm_flicker'] + df['norm_warping']) / 3)*0.5)

    df['performance_score'] = 1 - (df['generation_time'] - df['generation_time'].min()) / (df['generation_time'].max() - df['generation_time'].min() + 1e-10)
    df['overall_score'] = df['quality_score'] * 0.7 + df['performance_score'] * 0.3

    # Drop strength and deduplicate by taking the best overall_score
    group_cols = ['model', 'sampler', 'steps', 'frames', 'width', 'height']
    agg_df = df.groupby(group_cols, as_index=False).agg({
        'piqe': 'mean',
        'ssim': 'mean',
        'optical_flow_consistency': 'mean',
        'flicker_index': 'mean',
        'generation_time': 'mean',
        'quality_score': 'mean',
        'performance_score': 'mean',
        'overall_score': 'max'  # Keep the best
    })

    # Take top 10
    top_configs = agg_df.sort_values('overall_score', ascending=False).head(10)

    # Plot the top configurations
    plt.figure(figsize=(14, 10))
    y_pos = np.arange(len(top_configs))

    config_labels = [
        f"Sampler: {row['sampler']}, Steps: {row['steps']}, H: {row['height']}, W: {row['width']}"
        for _, row in top_configs.iterrows()
    ]

    bars = plt.barh(y_pos, top_configs['overall_score'], align='center')
    plt.yticks(y_pos, config_labels)
    plt.xlabel('Overall Score')
    plt.title('Top 10 Configurations by Overall Score')

    for i, (_, row) in enumerate(top_configs.iterrows()):
        plt.text(
            row['overall_score'] + 0.01,
            i,
            f"Qual: {row['quality_score']:.2f}, Perf: {row['performance_score']:.2f}",
            va='center'
        )

    plt.savefig(os.path.join(output_dir, "best_configurations.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Save to CSV
    top_configs.to_csv(os.path.join(output_dir, "top_configurations.csv"), index=False)


# 6. Add a correlation matrix to understand relationships between metrics
def create_correlation_analysis(df):
    # Select the metrics and parameters to analyze
    df['resolution'] = df['width'] * df['height']
    analysis_cols = quality_metrics + motion_metrics + performance_metrics + [
        p for p in ['steps', 'frames', 'resolution'] if p in df.columns and pd.api.types.is_numeric_dtype(df[p])
    ]
    
    # Create correlation matrix
    plt.figure(figsize=(12, 10))
    corr_matrix = df[analysis_cols].corr()
    
    # Plot correlation heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        center=0,
        fmt='.2f'
    )
    plt.title('Correlation Matrix Between Metrics and Parameters')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300)
    plt.close()
    
    # Also create a pairs plot for key metrics
    # Limit to avoid overcrowding - just use a subset of most important metrics
    key_metrics = ['piqe', 'ssim', 'flicker_index', 'generation_time']
    if 'steps' in df.columns and pd.api.types.is_numeric_dtype(df['steps']):
        key_metrics.append('steps')
    # if 'strength' in df.columns and pd.api.types.is_numeric_dtype(df['strength']):
    #     key_metrics.append('strength')
    
    # Create the pairs plot with sampler as hue
    plt.figure(figsize=(14, 12))
    pairs = sns.pairplot(df[key_metrics + ['sampler']], hue='sampler', palette='viridis')
    plt.suptitle('Relationships Between Key Metrics', y=1.02)
    plt.savefig(os.path.join(output_dir, "metrics_pairs_plot.png"), dpi=300, bbox_inches='tight')
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

    plt.title('Motion Magnitude Histogram Comparison by Frames', fontsize=16)
    plt.xlabel('Motion Magnitude Bucket', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
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
                label=f"{sampler}, Resolution: {resolution}"
            ))

    plt.title('Motion Magnitude Histogram Comparison by Resolution', fontsize=16)
    plt.xlabel('Motion Magnitude Bucket', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(hist_data)), range(len(hist_data)))
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "motion_histogram_by_resolution.png"), dpi=300, bbox_inches='tight')
    plt.close()

# # Create sample data (replace with your actual data)
# df = create_sample_data()

# # Rename piqe if needed (you had 'niqe' in your example)
# if 'niqe' in df.columns and 'piqe' not in df.columns:
#     df = df.rename(columns={'niqe': 'piqe'})
# 1. QUALITY METRICS DASHBOARD
def create_quality_metrics_dashboard(df):
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.8, wspace=0.3)

     # 1. CLIP Score by Sampler
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
    ax1.set_title('CLIP Score by Sampler', fontsize=14)
    ax1.set_xlabel('Sampler', fontsize=12)
    ax1.set_ylabel('CLIP Score (higher is better)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)

    # 2. PIQE by Sampler (Bar)
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
    ax2.set_title('PIQE by Sampler', fontsize=14)
    ax2.set_xlabel('Sampler', fontsize=12)
    ax2.set_ylabel('PIQE (lower is better)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)

    # 3. SSIM by Sampler (Bar)
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
    ax3.set_title('SSIM by Sampler', fontsize=14)
    ax3.set_xlabel('Sampler', fontsize=12)
    ax3.set_ylabel('SSIM (higher is better)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)

    # 4. Composite Quality Score by Resolution and Sampler
    ax4 = fig.add_subplot(gs[1, 1])
    df['resolution'] = df['width'].astype(str) + 'x' + df['height'].astype(str)

    metrics_by_res = df.groupby(['sampler', 'resolution']).agg({
        'clip_score': 'mean',
        'piqe': 'mean',
        'ssim': 'mean'
    }).reset_index()

    # Normalize metrics
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min()) if series.max() != series.min() else 0

    metrics_by_res['normalized_clip'] = normalize(metrics_by_res['clip_score'])
    metrics_by_res['normalized_ssim'] = normalize(metrics_by_res['ssim'])
    metrics_by_res['normalized_piqe'] = 1 - normalize(metrics_by_res['piqe'])  # invert PIQE

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
    ax4.set_title('Composite Quality Score by Resolution and Sampler', fontsize=14)
    ax4.set_xlabel('Resolution', fontsize=12)
    ax4.set_ylabel('Quality Score (higher is better)', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(title='Sampler', bbox_to_anchor=(1.01, 1), loc='upper left')

    # Manual layout adjust instead of tight_layout()
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig(os.path.join(output_dir, "quality_metrics_dashboard.png"), dpi=300, bbox_inches='tight')
    plt.close()



# 2. MOTION METRICS DASHBOARD
def create_motion_metrics_dashboard(df):
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.6, wspace=0.3)  # Increased hspace for more row spacing
    
    # 1. Optical Flow Consistency by Sampler
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
    ax1.set_title('Optical Flow Consistency by Sampler and Steps', fontsize=14)
    ax1.set_xlabel('Sampler', fontsize=12)
    ax1.set_ylabel('Optical Flow Consistency (higher is better)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='Steps', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # 2. Warping Error by Sampler
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
    ax2.set_title('Warping Error by Sampler and Steps', fontsize=14)
    ax2.set_xlabel('Sampler', fontsize=12)
    ax2.set_ylabel('Warping Error (lower is better)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Steps', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # 3. Flicker Index by Sampler
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
    ax3.set_title('Flicker Index by Sampler and Steps', fontsize=14)
    ax3.set_xlabel('Sampler', fontsize=12)
    ax3.set_ylabel('Flicker Index (lower is better)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(title='Steps', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # 4. Motion Magnitude Histograms by Sampler
    ax4 = fig.add_subplot(gs[1, 1])
    
    samplers = df['sampler'].unique()
    colors = plt.cm.plasma(np.linspace(0, 1, len(samplers)))
    
    for i, sampler in enumerate(samplers):
        sampler_df = df[df['sampler'] == sampler]
        
        # Average histograms for this sampler
        avg_hist = np.mean([row['motion_magnitude_histogram'] for _, row in sampler_df.iterrows()], axis=0)
        
        ax4.plot(
            range(len(avg_hist)),
            avg_hist,
            label=sampler,
            color=colors[i],
            linewidth=2
        )
    
    ax4.set_title('Average Motion Magnitude Histogram by Sampler', fontsize=14)
    ax4.set_xlabel('Motion Magnitude Bucket', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # 5. Composite Motion Quality Score
    ax5 = fig.add_subplot(gs[2, 0:])  # Stretching over 2 columns
    
    # Normalize and combine motion metrics
    motion_df = df.copy()
    motion_df['norm_flow'] = (motion_df['optical_flow_consistency'] - motion_df['optical_flow_consistency'].min()) / (motion_df['optical_flow_consistency'].max() - motion_df['optical_flow_consistency'].min())
    motion_df['norm_warp'] = 1 - (motion_df['warping_error'] - motion_df['warping_error'].min()) / (motion_df['warping_error'].max() - motion_df['warping_error'].min())
    motion_df['norm_flicker'] = 1 - (motion_df['flicker_index'] - motion_df['flicker_index'].min()) / (motion_df['flicker_index'].max() - motion_df['flicker_index'].min())
    
    # Composite score (higher is better)
    motion_df['motion_score'] = (motion_df['norm_flow'] + motion_df['norm_warp'] + motion_df['norm_flicker']) / 3
    
    # Group by frames and sampler
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
    
    ax5.set_title('Motion Quality Score by Number of Frames and Sampler', fontsize=14)
    ax5.set_xlabel('Number of Frames', fontsize=12)
    ax5.set_ylabel('Motion Quality Score (higher is better)', fontsize=12)
    ax5.grid(True, alpha=0.3)
    ax5.legend(bbox_to_anchor=(1.01, 0.5), loc='center left')
    
    plt.subplots_adjust(right=0.85, top=0.9)
    plt.savefig(os.path.join(output_dir, "motion_metrics_dashboard.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_dashboard(df):
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # --- 1. Generation Time vs. Number of Frames (per sampler) ---
    # ax1 = fig.add_subplot(gs[0, 0])
    # sns.lineplot(
    #     data=df,
    #     x='frames',
    #     y='generation_time',
    #     hue='sampler',
    #     style='sampler',
    #     markers=True,
    #     palette='inferno',
    #     ax=ax1
    # )
    # ax1.set_title('Generation Time vs. Number of Frames', fontsize=14)
    # ax1.set_xlabel('Number of Frames', fontsize=12)
    # ax1.set_ylabel('Generation Time (seconds)', fontsize=12)
    # ax1.grid(True, alpha=0.3)
    # ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    # --- 2. Generation Time vs. Number of Steps (per sampler) ---
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
    ax2.set_title('Generation Time vs. Number of Steps', fontsize=14)
    ax2.set_xlabel('Number of Steps', fontsize=12)
    ax2.set_ylabel('Generation Time (seconds)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    # --- 3. Generation Time by Resolution + Frames + Sampler (barplot) ---
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
    ax3.set_title('Generation Time by Resolution + Frames + Sampler', fontsize=14)
    ax3.set_xlabel('Resolution + Frames', fontsize=12)
    ax3.set_ylabel('Generation Time (s)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    # --- 4. Mean Generation Time by Sampler & Frames (lineplot) ---
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
    ax4.set_title('Generation Time by Sampler & Frames', fontsize=14)
    ax4.set_xlabel('Frames', fontsize=12)
    ax4.set_ylabel('Mean Generation Time (s)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.subplots_adjust(right=0.85, top=0.9)
    plt.savefig(os.path.join(output_dir, "performance_dashboard.png"), dpi=300, bbox_inches='tight')
    plt.close()


# Infer which parameters vary
potential_vars = ['steps', 'frames', 'seed', 'strength', 'sampler', 'width', 'height']
varying_params = [param for param in potential_vars if df[param].nunique() > 1]
# varying_params= ['steps', 'frames', 'seed', 'strength', 'sampler', 'width', 'height']
# Run all analyses
# print("Creating quality-performance tradeoff analysis...")
# create_quality_performance_tradeoff(df)

# print("Creating parameter impact analysis...")
# create_parameter_impact_analysis(df, varying_params)

# print("Creating motion analysis...")
# create_motion_analysis(df)

# print("Creating motion analysis2...")
# create_motion_analysis2(df)
# print("Creating motion analysis by frames...")
# create_motion_analysis_by_frames(df)
# print("Creating motion analysis by resolution...")
# create_motion_analysis_by_resolution(df)

# print("Creating comparative dashboard...")
# create_comparative_dashboard(df)

# print("Creating best configuration analysis...")
# create_best_configuration_analysis(df)

# print("Creating correlation analysis...")
# create_correlation_analysis(df)

# print("Creating quality vs motion stability based on sampler...")
# plot_quality_vs_motion_stability_by_sampler(df)
# Run all dashboard creations
print("Creating quality dashboard...")
create_quality_metrics_dashboard(df)
print("Creating metrics dashboard...")
create_motion_metrics_dashboard(df)
print("Creating performance dashboard...")
create_performance_dashboard(df)
print("Analysis complete! All graphs saved to:", output_dir)


