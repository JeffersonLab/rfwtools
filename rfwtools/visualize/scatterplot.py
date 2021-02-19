import matplotlib.pyplot as plt
import seaborn as sns
import rfwtools.visualize as viz


def scatterplot(data, x, y, title=None, figsize=None, drop_categories=True, **kwargs):
    """Creates/displays a single standard scatterplot.  Has extended marker set by default and external legend.

    Args:
        data (DataFrame) - A pandas DataFrame containing the data used in the scatter plot
        x (str) - The column name of df that holds the x values of the scatterplot
        y (ste) - The column name of df that holds the y values of the scatterplot
        title (str) - Title applied to plot (if not None)
        figsize (2-tuple(float)): The two dimensions of the size of the figure.  Passed to plt.figure.
        drop_categories (bool): Should unused categories be dropped from the hue and style columns (if categories)
        **kwargs (dict) - All remaining named parameters are passed to seaborn.scatterplot
    """

    # Create a figure and draw the plot
    if figsize is None:
        plt.figure()
    else:
        plt.figure(figsize=figsize)

    # Try to set some standardized marker sets
    if "style" in kwargs.keys():
        if 'markers' not in kwargs.keys():
            if kwargs['style'] == "cavity_label":
                kwargs['markers'] = viz.cavity_markers
            elif kwargs['style'] == "fault_label":
                kwargs['markers'] = viz.fault_markers
            elif kwargs['style'] == "zone_label":
                kwargs['markers'] = viz.zone_markers

    # Did the user not want to keep unused categories in the plot?
    dat = data.copy()
    if drop_categories:
        # Drop the the unused style categories
        if "style" in kwargs.keys():
            if dat[kwargs["style"]].dtype.name == "category":
                dat[kwargs["style"]] = dat[kwargs["style"]].cat.remove_unused_categories()

        # Drop the the unused hue categories
        if "hue" in kwargs.keys():
            if dat[kwargs["hue"]].dtype.name == "category":
                dat[kwargs["hue"]] = dat[kwargs["hue"]].cat.remove_unused_categories()

    # Create the scatterplot
    sns.scatterplot(data=dat, x=x, y=y, **kwargs)

    # Add title if specified
    if title is not None:
        plt.title(title)

    # Add a legend if we have color or marker differences to explain
    if "hue" in kwargs.keys() or "style" in kwargs.keys():
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # Show the plot
    plt.show()
