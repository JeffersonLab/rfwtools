"""This module is a work in progress left here for future improvement.  It's should be a light wrapper on t-SNE."""

import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

from rfwtools.extractor.down_sample import down_sample_extractor


def do_tsne_work(datasource):
    """Performs the 'whole' job of PCA from feature extraction to plot.  Represents our standard procedure."""

    events = datasource.get_example_array()
    events[0].load_data()

    # Standardized the signals and put them into a single series
    print("Starting data retrieval and standardization")
    n = len(events)
    feature_set_filename = f"feature_set_n{n}_step16.bz2"

    if os.path.exists(os.path.join("processed-output", feature_set_filename)):
        print("Loading saved feature set")
        datasource.load_feature_set(feature_set_filename)
    else:
        print("Extracting features")
        # ds.produce_feature_set(down_sample_extractor, max_workers=16, verbose=True)
        datasource.produce_feature_set(down_sample_extractor, max_workers=8, verbose=True)
        datasource.save_feature_set(feature_set_filename)

    datasource.feature_set.rename(columns={"fault-label": "fault_label", "cavity-label": "cavity_label"}, inplace=True)
    datasource.feature_set = datasource.feature_set.astype(
        {'fault_label': 'category', 'cavity_label': 'category', 'zone': 'category'})

    n_iters = [250, 500, 1000]
    perplexities = [2, 10, 30, 50, 70, 90, 100]
    # n_iters = [80000, 160000]
    # perplexities = [2, 10, 30, 50, 70, 90, 100]
    # n_iters = [250, 500, 2500, 5000, 10000, 20000, 40000, 80000, 160000]
    # perplexities = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # n_iters = [250, 260, 270]
    # perplexities = [2, 20]

    # Run the t_SNE jobs out to several points using a few different perplexities
    results = do_tsne_jobs(datasource.feature_set, n_iters, perplexities)

    # Plot these results
    plot_tsne_results(results, n_iters, perplexities)


def do_tsne_jobs(feature_df, n_iters, perplexities, metadata_cols=['zone', 'timestamp', 'fault_label', 'cavity_label']):
    y = feature_df[metadata_cols].copy()
    x = feature_df.drop(metadata_cols, axis=1)

    len_p = len(perplexities)
    len_n = len(n_iters)

    print("Computing t-SNE results.  Starting with PCA.")
    # Do PCA to get the dimensionality down to something reasonable for t-SNE
    pca = PCA(n_components=50)
    principal_components = pca.fit_transform(x)
    results = {}
    for i in range(len_n):
        n_iter = n_iters[i]
        for j in range(len_p):
            perplexity = perplexities[j]
            if n_iters[i] not in results:
                results[n_iters[i]] = {}
            # Now try t-SNE
            results[n_iter][perplexity] = do_tsne(principal_components, y, perplexity=perplexity, n_iter=n_iter)

    return results


def plot_tsne_results(results, perplexities, n_iters, step_size):
    len_p = len(perplexities)
    len_n = len(n_iters)

    n_plots = len_p * len_n
    print(f"Plotting t-SNE results in {n_plots} plots")

    for hue_by in ['fault_label', 'cavity_label', 'cf_label']:
        # Fig size works well as 10 * dim(n_iters), 6 * dim(perplexities) + 1
        plt.subplots(len_n, len_p, sharex="all", sharey='all', figsize=(len_n * 9, (len_p + 1) * 6))
        for i in range(len_p):
            perplexity = perplexities[i]
            for j in range(len_n):
                n_iter = n_iters[j]
                plot = i * len_n + (j + 1)
                ax = plt.subplot(len_p, len_n, plot)
                tsne_df = results[n_iter][perplexity]
                tsne_df['cf_label'] = pd.Series(
                    data=(tsne_df.fault_label.astype(str) + " c" + tsne_df.cavity_label.astype(str)), dtype='category')

                # Show a legend if this is the last plot
                plot_tsne(tsne_df=tsne_df, ax=ax, perplexity=perplexity, step_size=step_size, n_iter=n_iter,
                          hue_by=hue_by, legend=(plot % len_n == 0))
        plt.show()
        print(f"Displayed t-SNE grid with {hue_by} coloring")


def do_tsne(pc, y, perplexity, n_iter=5000):
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(pc)
    tsne_df = pd.concat((pd.DataFrame({
        'ts1': tsne_results[:, 0],
        'ts2': tsne_results[:, 1]
    }), y), axis=1)
    return tsne_df


def plot_tsne(tsne_df, ax, perplexity, step_size, n_iter=5000, hue_by="fault_label", legend=False):
    title = f"p={perplexity}, it={n_iter},n={tsne_df.shape[0]}, ds=1:{step_size}"
    if not legend:
        sns.scatterplot(tsne_df.ts1, tsne_df.ts2, hue=tsne_df[hue_by], alpha=0.7, legend=False)
    else:
        sns.scatterplot(tsne_df.ts1, tsne_df.ts2, hue=tsne_df[hue_by], alpha=0.7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_title(title)
