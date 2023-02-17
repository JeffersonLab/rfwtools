.. rfwtools documentation master file, created by
   sphinx-quickstart on Mon Feb 22 16:23:19 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to rfwtools's documentation!
====================================

This package aims to provide standardized and easy usage of the C100's harvester RF fault waveform data.

Github Page: https://github.com/JeffersonLab/rfwtools

Contents:

.. autosummary::
  :toctree: _autosummary
  :template: custom-module-template.rst
  :recursive:

  rfwtools

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Usage Examples:
-----------------
Here are a couple of different workflows supported by the package.

Initial Setup
~~~~~~~~~~~~~~~~~~~
Start by saving this data in my-sample-labels.txt in the Config().label_dir directory (defaults to ./data/labels/).
**THESE FIELDS SHOULD BE TAB SEPARATED.  DOCUMENTATION SYSTEM INSISTS ON CONVERTING THEM TO SPACES.  PLEASE FIX IF YOU
TRY THIS EXAMPLE ON YOUR OWN**

::

    zone	cavity	cav#	fault	time
    1L25	4	44	Microphonics	2020/03/10 01:08:41
    2L24	5	77	Controls Fault	2020/03/10 01:42:03
    1L25	5	45	Microphonics	2020/03/10 02:50:07
    2L26	8	96	E_Quench	2020/03/10 02:58:13
    1L25	5	45	Microphonics	2020/03/10 04:55:21
    1L22	4	20	Quench_3ms	2020/03/10 05:06:13
    1L25	5	45	Microphonics	2020/03/10 07:35:32
    2L22	0	57	Multi Cav turn off	2020/03/10 07:59:49
    2L23	0	65	Multi Cav turn off	2020/03/10 07:59:56
    2L24	0	73	Multi Cav turn off	2020/03/10 08:00:03

You may also need to create a configuration file or update the configuration in code.  Place a file called rfwtools.cfg
in your current directory with the following information.  Other options are available.:

::

   data_dir: /path/to/parent_dir/of/zone_dirs
   label_dir: /path/to/dir/containing/label_files
   output_dir: /path/to/where/save/files/live

Alternatively, do this in code:

::

   from rfwtools.config import Config

   Config().data_dir = "/path/to/parent_dir/of/zone_dirs"
   Config().label_dir = "/path/to/dir/containing/label_files"
   Config().output_dir = "/path/to/where/save/files/live"

Workflow Using DataSet and "regular" Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here we use the tiny my-sample-labels.txt setup above.  Much the workflow goes through the DataSet object.  Since
the Example class is the default for the DataSet, you don't need to specify much.

::

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

Workflow Using DataSet and WindowedExamples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is very similar to the above, except that now we are using WindowedExamples and everything that comes with it.
Please note that this is one of two ways to get "windowed" data.  The other is to use the
rfwtools.extractor.window_extractor method (see rfwtools.extractor.windowing for details).

::

   from rfwtools.data_set import DataSet
   from rfwtools.extractor.autoregressive import autoregressive_extractor
   from rfwtools.example import ExampleType
   from rfwtools.example_validator import WindowedExampleValidator

   # This tells the DataSet that you will want to work with WindowedExamples
   e_type = ExampleType.WINDOWED_EXAMPLE

   # These parameters will be passed to the Example objects upon construction, e.g., all example will have the same
   # window.  Here we assume 0.2ms sample steps, and we want windows of 100ms, so 100*(1/0.2) = 500.
   e_kw = {"start": -1536, "n_samples": 500}

   # The WindowedExample class works slightly differently so it needs a different validator.  This makes sure that the
   # each example has all of the characteristics we want (sample step size, number of capture files, etc.).
   ev = WindowedExampleValidator()

   # Create a DataSet.  For demo-purposes, I would make a small label file and run through.  This can take hours/days to
   # process all of our data.
   ds = DataSet(label_files=['my-sample-labels.txt'], e_type=e_type, example_validator=ev, example_kwargs=e_kw)

   # This will process the label files you have and create an ExampleSet under ds.example_set
   ds.produce_example_set()

   # From here on it's the same
   ...

Workflow Without Using a DataSet:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There may be times when using a DataSet is cumbersome.  A DataSet is really useful for generating an ExampleSet and/or
FeatureSet, but their is no need to use one if you already have saved files ready to load.

Here we load a file and add a day of week to the ExampleSet:

::

   from rfwtools.example_set import ExampleSet

   es = ExampleSet()
   es.load_csv("my_example_set.csv")
   df = es.get_example_df()
   df['my_feature'] = df.dtime.dt.day_name()
   es.update_example_set(df)

Here we determine bypassed cavity information for a FeatureSet:

::

   from rfwtools.feature_set import FeatureSet
   from rfwtools.example import Example
   import pandas as pd

   # This method determines if a cavity was producing gradient above a threshold.  It not, it is considered bypassed.
   def bypassed_cavity_extractor(example: Example, threshold: float = 0.5) -> pd.DataFrame:

       example.load_data()
       df = example.event_df
       example.unload_data()

       out = pd.DataFrame(
           {'has_bypassed': [False], 'num_bypassed': [0], 'c1_bypassed': [False], 'c2_bypassed': [False], 'c3_bypassed': [False],
            'c4_bypassed': [False], 'c5_bypassed': [False], 'c6_bypassed': [False], 'c7_bypassed': [False],
            'c8_bypassed': [False]
            })

       for cav in range(1,9):
           if df[f"{cav}_GMES"].max() < threshold:
               out.has_bypassed = True
               out.num_bypassed += 1
               out[f"c{cav}_bypassed"] = True

       return out

   # Load up the FeatureSet
   fs = FeatureSet()
   fs.load_csv("my_feature_set.csv")

   # Add the bypassed column data to the DataFrame
   df = fs.get_example_df()
   bypassed_df = pd.concat(df['example'].apply(bypassed_cavity_extractor).values, ignore_index=True)
   df = pd.concat([df, bypassed_df], axis=1)

   # Update the FeatureSet
   new_cols = ['has_bypassed', 'num_bypassed', 'c1_bypassed', 'c2_bypassed', 'c3_bypassed',  'c4_bypassed',
               'c5_bypassed', 'c6_bypassed', 'c7_bypassed', 'c8_bypassed']
   m_cols = fs.metadata_columns + new_cols
   fs.update_example_set(df, metadata_columns=m_cols)
   fs.do_pca_reduction()
   fs.display_2d_scatterplot(style='zone', hue='num_bypassed')
