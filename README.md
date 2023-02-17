# rfwtools

This package provides commonly used functionality around CEBAF C100 RF Waveforms collected by the JLab harvestser. This
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

Internally the package leverages a Config class that contains directory locations, URLs for network services, etc.. On
first reference, this class looks for and parses a config file, ./rfwtools.cfg. Below is simplified example file.

```yaml
data_dir: /some/path/rfw-research/data/waveforms/data/rf
label_dir: /some/path/rfw-research/data/labels
output_dir: /some/path/rfw-research/processed-output
```

data_dir
: Base directory containing RF waveform data directory structures (i.e., directory containing zone directories). This
path may include a symlink on Linux if you do not wish to duplicate data. The path structure should mimic that found in
opsdata.
label_dir
: Directory contain label files (typically provided by Tom Powers)
output_dir
: Default directory for writing/reading saved files and other processed output

If no file is found, file system paths are relative the project base, which is assumed to be the current working
directory. You can adjust these parameters in code as in the example below.

```python
from rfwtools.config import Config
Config().data_dir = "/some/new/path"
```

## Usage
Previous usage of this was to download a template directory structure with source code. This proved cumbersome, and
did not result in widespread usage. Below is a simple example that assume the above locations were sensibly defined.
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

## Developer Notes

Here are some notes on the development process.

First clone the repo. Then create a venv for development.

```bash
git clone https://github.com/JeffersonLab/rfwtools
python3.7 -m venv venv
```

Activate the venv and install the development requirements. These packages are used strictly in packaging, deploying,
and testing

```bash
source venv/bin/activate.csh
pip3 install -r requirements-dev.txt
```

Now you can build wheels and source distributions, run unit tests, and upload to the test PyPI or PyPI. One thing
I like to do is create a project in a different directory and then install this package in editable mode. Instead
of actually installing it, pip creates a symlink back to your package directory and your source changes are reflected
in the downstream project without reinstalling. You do have to re-import packages or restart your interpreter though.

```bash
mkdir /some/other/my_project
cd /some/other/my_project
python -m venv venv
source venv/bin/activate.csh
pip install -e /path/to/rfwtools
```

If you want to make source changes then you will need to install the packages in requirements.txt. The versions
listed where the ones used in the last development cycle. You may want to update those versions, but make sure to
test!

```bash
pip install -r requirements.txt
```

To run a unittests in multiple environments. Windows and linux have slightly different configurations. These match
the environment lists.

```bash
tox -e py37-windows
tox -e py37-linux
```

To run them directly in an IDE with builtin test runner, do the equivalent of this.

```bash
cd /path/to/.../rfwtools
python3 -m unittest
```

To build documentation that can be used in github.

From windows:

```PowerShell
cd docsrc
.\make.bat github
git add .
git commit -m"Updated documentation"
```

From Linux:

```bash
docsrc/build-docs.bash
git add .
git commit -m"Updated documentation"
```

You should increment version numbers in setup.cfg and put out a new package to PyPI once a release is ready(shown below)
.
Update the requirements files if they changed. At a minimum, this should always be requirements.txt. See comments
below for details.

```bash
pip freeze > requirements.txt
```

To build distributions. You may need to remove directory content if rebuilding

```bash
python setup.py sdist bdist_wheel
```

To upload to the test PyPI repo. You may need to add the `--cert /etc/pki/tls/cert.pem` option for SSL problems.
Make sure to edit setup.cfg with latest info as shown below using vi and have built the package.

```csh
vi setup.cfg
source venv/bin/activate.csh
python setup.py sdist bdist_wheel
twine upload --repository testpypi dist/*
```

To upload to the production PyPI repo. First edit setup.cfg with latest info.

```bash
twine upload --repository pypi dist/*
```

To install from production PyPI:

```bash
pip install rfwtools
```

To install from Test PyPI:

```bash
pip3 install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ rfwtools
```

### Additional Developer Comments

requirements.txt are the versions that were used from within my IDE and with my IDE unit test runner. This is the set
that worked during development, installation set will probably be different. This is every installed package, not just
the ones that my package directly uses.

requirements-dev.txt are the versions of tools required to build, test, and distribute the package. These are the set
that worked during the last development cycle.

requirements-testing.txt are the packages that needed to be installed for testing to work. They are basically the same
as requirements.txt, but with a few extras used exclusively in tests and the local rfwtools package itself.

### Certified Notes

The process for certified installation are largely captured in the `setup-certified.bash` script. Most of the basic
developer process is the same, but you will need to run through the certified installation process completely to make
sure that everything works as expected. At the end of this process you will have dropped the package files in a
directory.  That's all the get installed in the certified area.

1. Generate a certified tarball once you think development is done.  
   ```./setup-certified tarball rfwtools<version>```
2. Copy this tarball to a temp directory and unzip it.
   ```csh
   cd ..
   mkdir tmp
   mv rfwtools<version>.tar.gz tmp
   cd tmp
   tar -xzf rfwtools<version>.tar.gz
   cd rfwtools<version>
   ```
3. Now run through the standard process described by `setup-certified.bash -h`. Make sure to review the docs directory
   when done.This is something like the following:
   ```bash
   ./setup-certified.bash test
   ./setup-certified.bash docs
   ./setup-certified.bash build
   ```
4. You can also test the final installation if you have a target directory ready. You should find some wheel or tar.gz
   files in the target directory when done.
   ```bash
   mkdir -p /tmp/pretend-certified/rfwtools/<version>
   ./setup-certified.bash install /tmp/pretend-certified/rfwtools/<version>
   ```
5. Now compact the tarball to ensure that the to-be-archived code is what you want.
   ```bash
   ./setup-certified.bash compact
   ```