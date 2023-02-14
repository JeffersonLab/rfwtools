"""The module provides some customized heatmap visualizations for fault data."""

import copy
import math
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns


def heatmap_cavity_vs_fault_label_counts(data: pd.DataFrame, title: str = None, vmin: float = None, vmax: float = None,
                                         margins: bool = False) -> None:
    """Displays a heat map plot of counts of cavity/fault label pairs

    Arguments:
        data: The DataFrame to use.  Should conform to ExampleSet standards.
        title: The chart title
        vmin, vmax: seaborn.heatmap parameters.  Values to anchor the colormap.  If None, drawn from data
        margins: pd.pivot_table parameter.  Should row and column sums be included.
    """
    # Create a pivot table DataFrame that is a matrix with values that are counts with cavity/fault labels as
    # columns/rows.  values=dtime here is an arbitrary choice of something to count.
    hm_df = pd.pivot_table(data=data, values='dtime', columns='cavity_label', margins=margins, index='fault_label',
                           aggfunc='count').fillna(0)

    if vmax is None:
        # Get the max value ignoring the (sub)totals
        vmax = np.max(np.max(hm_df.iloc[:-1, :-1]))

    # Generate the heatmap
    plt.figure()
    ax = sns.heatmap(data=hm_df, cmap="Blues", annot=True, fmt='g', vmin=vmin, vmax=vmax)

    # Title the plot with a sane default if nothing is supplied.
    if title is None:
        zones = data.zone.unique().join(", ")
        start = data['dtime'].min().strftime("%Y-%m-%d %H:%M:%S")
        end = data['dtime'].max().strftime("%Y-%m-%d %H:%M:%S")
        title = f"{zones}\n{start} - {end}"
    ax.set_title(title)

    # Show the plot
    plt.subplots_adjust(top=0.85, bottom=0.15, right=0.95, left=0.3)
    plt.show()


def show_fault_cavity_count_by_zone(data: pd.DataFrame, zones: List[str], dt_breaks: List[datetime] = None) -> None:
    """Creates a set of grids of heat maps (one per zone) which shows the count of fault/cavity combinations.

    Arguments:
        data:
            DataFrame containing fault information.  Required columns are 'fault_label', 'cavity_label', 'zone',
            (all category dtype), and 'dtime' of type datetime
        zones:
            A list of strings of the zones to show in the heat map grid
        dt_breaks:
            A list of datetime objects to use a break points in a series of produced plots
    """

    if dt_breaks is None:
        _show_fault_cavity_count_by_zone(data, zones)
    else:

        # Make sure we have a list-like structure
        if not isinstance(dt_breaks, (list, tuple)):
            dt_breaks = [dt_breaks]

        # Make sure we get these in order
        dt_breaks.sort()

        # Split up the breaks into filter points
        dt_ranges = []
        prev = None
        for i in range(0, len(dt_breaks)):
            curr = dt_breaks[i]
            if prev is None:
                prev = datetime.strptime("1970-01-01", "%Y-%m-%d")
            dt_ranges.append((prev, curr))
            prev = curr
        dt_ranges.append((prev, datetime.now()))

        for start, end in dt_ranges:
            dat = data[start < data['dtime']]
            dat = dat[dat['dtime'] <= end]

            fmt = '%Y-%m-%d %H:%M:%S'
            title = start.strftime(fmt) + " - " + end.strftime(fmt)
            _show_fault_cavity_count_by_zone(data=dat, zones=zones, title=title)


def _show_fault_cavity_count_by_zone(data: pd.DataFrame, zones: List[str], title: str = None, nrows: int = 1,
                                     ncols: int = None, vmin: float = None, vmax: float = None):
    """Internal method for splitting up ExampleSet style DataFrame data into per zone subsets and displaying heatmaps"""
    # Figure out a dimension of the multiplot
    if ncols is None:
        ncols = math.ceil(len(zones) / nrows)

    # Prepare the data to be displayed
    counts = pd.pivot_table(data=data, index=['fault_label', 'cavity_label'], columns="zone", values="dtime",
                            aggfunc="count").fillna(0)

    counts.columns = list(str(x) for x in counts.columns)

    # Get the maximum, but we want it to be at least 10 in the case of all zeros.  Allow for user override
    if vmax is None:
        count_max = counts.max().max()
        if math.isnan(count_max) or count_max == 0:
            count_max = 10
    else:
        count_max = vmax
    if vmin is None:
        count_min = 0
    else:
        count_min = vmin

    # Create the subplot grid
    # fig, axn = plt.subplots(nrows, ncols, sharex="all", sharey="all", figsize=(20, 3))
    # The exact size to reserve for the figure is sort of a guess.  3x3 for each plot plus extra for cbar and y labels
    fig, axn = plt.subplots(nrows, ncols, sharex="all", sharey="all", figsize=(2 + 3 * ncols, 3 * nrows))

    if title is None:
        start = data['dtime'].min().strftime('%Y-%d-%m %H:%M:%S')
        end = data['dtime'].max().strftime('%Y-%d-%m %H:%M:%S')
        title = f"{start} - {end}"

    fig.suptitle(title)

    # Iterate over zones adding a plot as we go
    i = 1
    for zone in zones:
        if zone is None:
            continue
        if zone not in counts.columns:
            # Create a DataFrame (matrix) of zeros since the zone had no faults
            rows = len(data.fault_label.cat.categories)
            cols = len(data.cavity_label.cat.categories)
            hm_df = pd.DataFrame(data=np.zeros((rows, cols), dtype='float64'), index=data.fault_label.cat.categories,
                                 columns=data.cavity_label.cat.categories)

        else:
            # Create a DataFrame with the counts of fault/cavity pairs for this zone
            hm_df = pd.pivot_table(data=counts, index='fault_label', columns='cavity_label', values=zone).fillna(0)

        # ax = plt.subplot(1, len(zones), i)
        # ax = axn[math.floor(i / ncols), i % ncols]
        ax = plt.subplot(nrows, ncols, i)

        # cmap = "RdBu_r"
        # Some common values
        cmap = copy.copy(plt.cm.get_cmap("Blues"))
        cmap.set_under("white")
        cmap.set_over('red')
        annot = True
        xticklabels = True
        xlabel = "cavity label"

        # Draw the subplot as needed.  First and last are different
        if i % ncols == 1:
            sns.heatmap(hm_df, cmap=cmap, vmin=count_min, vmax=count_max, annot=annot, fmt='g', cbar=False,
                        xticklabels=xticklabels)
            ax.set_ylabel('fault label')
        elif i % ncols == 0:
            sns.heatmap(hm_df, cmap=cmap, vmin=count_min, vmax=count_max, annot=annot, fmt='g', cbar=True,
                        yticklabels=False, xticklabels=xticklabels)
        else:
            sns.heatmap(hm_df, cmap=cmap, vmin=count_min, vmax=count_max, annot=annot, fmt='g', cbar=False,
                        yticklabels=False, xticklabels=xticklabels)

        # Add labels, etc.
        ax.set_title(zone)
        ax.set_xlabel(xlabel)
        i += 1

    # In case we didn't fill up the whole array.  This isn't perfect, but
    while i <= ncols * nrows:
        ax = plt.subplot(nrows, ncols, i)
        ax.axis('off')
        i += 1

    # Display the plot
    plt.subplots_adjust(top=0.8, bottom=0.2, right=0.95)
    plt.show()
