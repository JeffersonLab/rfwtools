import os
import warnings

from rfwtools.config import Config
from rfwtools.data_set import DataSet
import pandas as pd


# Define the feature extractor function.
# This is a dummy function just returns the first two values of two signals
def my_extract_on_data(ex):
    """A simple extractor function for use with a DataSet.  This uses waveform data and slower due to disk reads."""
    ex.load_data()
    out = pd.DataFrame({'1_1_GMES': [ex.event_df['1_GMES'][0]], '1_1_PMES': [ex.event_df['1_PMES'][0]]})
    ex.unload_data()
    return out


def my_extract_no_data(ex):
    """A simple extractor function for use with a DataSet.  This does not use data from disk and is quicker"""
    out = pd.DataFrame({"linac": [float(ex.event_zone[0])],
                        "hour": [ex.event_datetime.hour],
                        "minute": [ex.event_datetime.minute],
                        "second": [ex.event_datetime.second]})
    return out


def main():
    # The test/test-labels/limited-real directory has downsampled versions of the real label files.
    # Changing the config to point to this test set.  Otherwise it points to data/labels
    Config().label_dir = os.path.join(Config().app_dir, "test", "test-labels", "limited-real")

    # Winter_2020_faults_limited.txt has only the first ~100 faults
    # ds = DataSet(label_files=['WINTER_2020_faults_limited.txt'])

    # To get all of the label files don't provide the label_files option
    ds = DataSet()

    # This will process all the specified label files and produce a set of examples.
    # Expect a lot of output here with report=True
    ds.produce_example_set(report=True)

    # Let's see how many events we ended up with after all of that.
    print("Example set size: {}".format(len(ds.example_set.get_example_df())))

    ### Let see some charts for this Example Set ###

    # Timeline of faults
    ds.example_set.display_timeline()

    # Cavity vs Fault label heatmaps
    ds.example_set.display_summary_label_heatmap()
    ds.example_set.display_zone_label_heatmap(zones=["1L22", "1L23", '1L24', '1L25', '1L26'])
    ds.example_set.display_zone_label_heatmap(zones=["2L22", "2L23", '2L24', '2L25', '2L26'])

    # Frequency barplots - lets see how many Examples we have through various lenses (cavity_label by fault_label,
    # label_source by zone, count by only label source, etc.)
    ds.example_set.display_frequency_barplot('label_source', 'fault_label')
    ds.example_set.display_frequency_barplot('label_source', 'cavity_label')
    ds.example_set.display_frequency_barplot('zone', 'cavity_label')
    ds.example_set.display_frequency_barplot('cavity_label', 'fault_label')
    ds.example_set.display_frequency_barplot('cavity_label')
    ds.example_set.display_frequency_barplot('cavity_label')

    # Print out some summary info about the example set.  Here we get a copy of the example set as a DataFrame.
    example_df = ds.example_set.get_example_df()

    print("\n=== DF describe ===")
    print(example_df.describe())

    print("\n=== entries per file ===")
    print(example_df.label_source.value_counts())

    # Do the feature extraction!  Note: works in parallel
    print("\n=== Produce Feature Set ===")
    # ds.produce_feature_set(my_extract_no_data)
    # Real feature extraction functions are defined under lib.extractors
    from rfwtools.extractor.autoregressive import autoregressive_extractor

    ds.produce_feature_set(autoregressive_extractor)

    # Lets do some dimensional reduction and visualization.  The report option prints out some info about the PCs.
    print("\n=== Dim Reduction ===")
    ds.feature_set.do_pca_reduction(report=True)

    # Now plot the first two primary components
    ds.feature_set.display_2d_scatterplot()

    # The DataSet will also download an ExampleSet from our web service that contains the results from our in service models
    # This behaves just like the label file based ExampleSet, but it should also have label confidence numbers.

    # Plot a frequency of cavity labels by label_source.  Here the source is the model name.
    ds.example_set_model.display_frequency_barplot("cavity_label", "label_source")

    # You can create a Classification report of one ExampleSet against another.  The "other" ExampleSet is considered ground
    # truth.  Now compare our online models against the label set.
    print("Intersection of all")
    ds.example_set_model.get_classification_report(ds.example_set, "cavity_label")

    print("Intersection of random_forest_v1_2 and all label files")
    ds.example_set_model.get_classification_report(ds.example_set, "cavity_label",
                                                   query="label_source == 'random_forest_v1_2'")

    return ds


if __name__ == "__main__":
    ds = main()
