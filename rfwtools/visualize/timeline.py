import matplotlib.pyplot as plt
import seaborn as sns

def swarm_timeline(data, **kwargs):
    """Generate and display a swarmplot of fault/cavity label pairs over time.  Extra kwargs passed to swarmplot.

    Note: kwargs especially usefull for modifying marker size with s=<int>.

    Args:
        data (DataFrame) - Expects a DataFrame with the format from ExampleSet
    """
    plt.figure(figsize=(20, 9))
    ax = sns.swarmplot(x=data.cavity_label, y=data['dtime'], hue=data.fault_label, **kwargs)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    plt.show()
