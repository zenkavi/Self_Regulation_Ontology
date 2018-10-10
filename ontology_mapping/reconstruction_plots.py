import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from selfregulation.utils.plot_utils import beautify_legend, format_num, save_figure

def plot_factor_reconstructions(reconstructions, title=None, size=12, 
                                filename=None, dpi=300):
    # construct plotting dataframe
    c = reconstructions.columns.get_loc('var') 
    ground_truth = reconstructions.query('label=="true"')
    ground_truth.index = ground_truth['var']
    ground_truth = ground_truth.iloc[:, :c]
    ground_truth.columns = [str(c) + '_GT' for c in ground_truth.columns]
    plot_df = reconstructions.query('label=="partial_reconstruct"') \
                .groupby(['pop_size', 'var']).mean() \
                .join(ground_truth)
    pop_sizes = sorted(reconstructions.pop_size.dropna().unique())
    # plot
    sns.set_context('talk')
    sns.set_style('white')
    f, axes = plt.subplots(c,len(pop_sizes),figsize=(size,size*1.2))
    colors = sns.color_palette(n_colors = len(pop_sizes))

    for j, pop_size in enumerate(pop_sizes):
        reconstruction = plot_df.query('pop_size == %s' % pop_size)
        for i, factor in enumerate(plot_df.columns[:c]):
            factor = str(factor)
            ax = axes[i][j]
            ax.scatter(reconstruction.loc[:,factor+'_GT'],
                       reconstruction.loc[:,factor],
                       color=colors[j],
                      s=size*1.5, alpha=.5)
            ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", zorder=-1)
            ax.tick_params(axis='both', labelleft=False, labelbottom=False, bottom=False, left=False)
            if j==(len(pop_sizes)-1) and i==0:
                ax.set_ylabel('Reconstruction', fontsize=size*1.5)
                ax.set_xlabel('Ground Truth', fontsize=size*1.5)
            if j==0:
                ax.set_ylabel(factor, fontsize=size*2)
            if i==(c-1):
                ax.set_xlabel(pop_size, fontsize=size*2)
            # indicate the correlation
            corr = reconstruction.corr().loc[factor, factor+'_GT']
            s = '$\it{r}=%s$' % format_num(corr)
            ax.text(.05,.85, s, transform = ax.transAxes, fontsize=size*1.5)
    f.text(0.5, 0.06, 'Subpopulation Size', ha='center', fontweight='bold', fontsize=size*2)
    f.text(0.04, 0.5, 'Factor', va='center', rotation=90, fontweight='bold', fontsize=size*2)
    if title:
        f.suptitle(title, y=.93, size=size*2, fontweight='bold')
    if filename is not None:
        save_figure(f, filename, {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()

def plot_reconstruction_hist(reconstructions, title=None, size=12, 
                             filename=None, dpi=300):
    sns.set_context('talk')
    sns.set_style('white')
    reconstructions = reconstructions.query('label == "partial_reconstruct"')
    pop_sizes = sorted(reconstructions.pop_size.dropna().unique())
    f, axes = plt.subplots(1,len(pop_sizes),figsize=(size,size/4))
    colors = sns.color_palette(n_colors = len(pop_sizes))
    for i, pop_size in enumerate(pop_sizes):
        reconstruction = reconstructions.query('pop_size == %s' % pop_size) \
                                         .groupby('var')['corr_score'].mean()
        reconstruction.hist(bins=20, ax=axes[i], grid=False, color=colors[i])
        axes[i].tick_params(length=1)
        axes[i].set_xlim(-.2,1)
        if i == 0:
            axes[i].set_ylabel('# of Variables', fontsize=size*1.5)
        for spine in ['top', 'right']:
            axes[i].spines[spine].set_visible(False)
        axes[i].set_title('Pop Size: %s' % int(pop_size), fontsize=size*1.5)
    f.text(0.5, -0.1, 'Average Reconstruction Score', 
           ha='center',  fontsize=size*1.5)
    if title:
        f.suptitle(title, y=1.15, size=size*1.75)
    if filename is not None:
        save_figure(f, filename, {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()