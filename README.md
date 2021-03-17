# rfwtools

This package provides commonly used functionality around CEBAF C100 RF Waveforms collected by the JLab harvestser.  This
includes data management such as download capture files, reading data from disk, parsing label files, running feature
extraction tasks, and generating data reports and visualizations.

## Latest API Documentation

https://jeffersonlab.github.io/rfwtools/

## Installation

This package has been posted to PyPI to ease installation.

```bash
pip install rfwtools
```

If you would rather edit the code while using it you should do a git clone to a local directory, then install that
package in edit-able mode.

```bash
cd /some/place
git clone https://github.com/JeffersonLab/rfwtools .

# Install the package (recommended that you use a virtual environment, etc.)
pip install -e /some/place/rfwtools
```

## Configuration

Internally the package leverages a Config class that contains directory locations, URLs for network services, etc..  On
first reference, this class looks for and parses a config file, ./rfwtools.cfg.  Below is simplified example file.

```yaml
data_dir: /some/path/rfw-research/data/waveforms/data/rf
label_dir: /some/path/rfw-research/data/labels
output_dir: /some/path/rfw-research/processed-output
```

data_dir
: Base directory containing RF waveform data directory structures (i.e., directory containing zone directories).  This path may include a symlink on Linux if you do not wish to duplicate data.  The path structure should mimic that found in opsdata.
label_dir
: Directory contain label files (typically provided by Tom Powers)
output_dir
: Default directory for writing/reading saved files and other processed output

If no file is found, file system paths are relative the project base, which is assumed to be the current working
directory.  You can adjust these parameters in code as in the example below.

```python
from rfwtools.config import Config
Config().data_dir = "/some/new/path"
```

## Usage
Previous usage of this was to download a template directory structure with source code.  This proved cumbersome, and 
did not result in widespread usage.  Below is a simple example that assume the above locations were sensibly defined.
It shows some of what you can accomplish with the package.

```python
from rfwtools.data_set import DataSet
from rfwtools.extractor.autoregressive import autoregressive_extractor

# Create a DataSet.  For demo-purposes, I would make a small label file and run through.  This can take hours/days to
# process all of our data
ds = DataSet(label_files=['my-sample-labels.txt'])

# This will process the label files you have and create an ExampleSet under ds.example_set
ds.produce_example_set()

# Save a CSV of the examples.
ds.save_example_set_csv("my_example_set.csv")

# Show data from label sources, color by fault_label
ds.example_set.display_frequency_barplot(x='label_source', color_by="fault_label")

# Show heatmaps for 1L22-1L26
ds.example_set.display_zone_label_heatmap(zones=['1L22', '1L23', '1L24', '1L25', '1L26'])

# Generate autoregressive features for this data set.  This can take a while - e.g. a few seconds per example.
ds.produce_feature_set(autoregressive_extractor)

# Save the feature_set to a CSV
ds.save_feature_set_csv("my_feature_set.csv")

# Do dimensionality reduction
ds.feature_set.do_pca_reduction(n_components=10)

# Plot out some different aspects
# Color by fault, marker style by cavity
ds.feature_set.display_2d_scatterplot(hue="fault_label", style="cavity_label")

# Color by zone, marker style by cavity, only microphonics faults
ds.feature_set.display_2d_scatterplot(hue="zone", style="cavity_label", query="fault_label == 'Microphonics'")
```
