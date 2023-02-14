import filecmp
import os
import unittest
from datetime import datetime

from unittest import TestCase
from rfwtools.feature_set import FeatureSet
import pandas as pd
import numpy as np
import math
import test


def make_dummy_df(n, standard=True):
    """Generates a dummy DataFrame with id as only metadtata column.  Optionally use standard (zone, etc.).

    Value of columns is not same as real data.
    """

    if standard:
        df = pd.DataFrame({'zone': pd.Categorical(['1L22'] * n),
                           "dtime": [datetime(year=2020, month=3, day=13)] * n,
                           'cavity_label': pd.Categorical(["1"] * n),
                           'fault_label': pd.Categorical(["Microphonics"] * n),
                           'cavity_conf': [math.nan] * n,
                           'fault_conf': [math.nan] * n,
                           'label_source': ["test_source"] * n,
                           'id': [x for x in range(n)],
                           "x1": [x for x in range(n)],
                           'x2': [1] * n,
                           'x3': [2] * n,
                           'x4': [3] * n
                           })
    else:
        df = pd.DataFrame({'id': [x for x in range(n)],
                           "x1": [x for x in range(n)],
                           'x2': [1] * n,
                           'x3': [2] * n,
                           'x4': [3] * n
                           })

    return df


class TestFeatureSet(TestCase):

    def test_construction(self):
        # Test a few basic constructions.
        df = make_dummy_df(3, standard=False)

        # Check that the missing metadata columns throws
        with self.assertRaises(ValueError):
            FeatureSet(df=df)

        # Construct a more standard looking one.
        df = make_dummy_df(3)

        # Check that we can supply custom metadata_columns values
        FeatureSet(df=df, metadata_columns=['id'])

        # Test that standard construction works
        FeatureSet(df)
        FeatureSet(df, name="testing!")

    def test_do_pca_reduction(self):
        # Test with a _very_ simple case
        # Create a dummy DataFrame with only one column having variation - should have only on non-zero PC
        df = make_dummy_df(3)
        fs = FeatureSet(df=df, metadata_columns=['id'])
        meta_df = fs.get_example_df()[['zone', 'dtime', 'cavity_label', 'fault_label', 'cavity_conf', 'fault_conf',
                                       'example', 'label_source', 'id']]

        # Defaults to three components.  Test the no standardization option
        fs.do_pca_reduction(standardize=False, report=False)
        exp = pd.DataFrame({"id": [0, 1, 2], "pc1": [1., 0., -1.], "pc2": [0., 0., 0.], "pc3": [0., 0., 0.]})
        exp = meta_df.merge(exp, on='id')
        self.assertTrue((exp.equals(fs.get_pca_df())), f"exp = \n{exp}\npca_df = \n{fs.get_pca_df()}")

        # Check that this works with standardization
        fs.do_pca_reduction(standardize=True, report=False)
        exp = pd.DataFrame({"id": [0, 1, 2], "pc1": [1.224744871391589, 0., -1.224744871391589], "pc2": [0., 0., 0.],
                            "pc3": [0., 0., 0.]})
        exp = meta_df.merge(exp, on='id')

        self.assertTrue((exp.equals(fs.get_pca_df())), f"exp = \n{exp}\npca_df = \n{fs.get_pca_df()}")

        # Check that the explained variance is what we expect (all on one PC)
        self.assertTrue((np.array([1., 0., 0.]) == fs.pca.explained_variance_ratio_).all())

    def test_eq(self):
        # Test our equality operator

        # Make some Feature Sets
        df = make_dummy_df(4, standard=True)

        # Identical
        fs1 = FeatureSet(df=df, metadata_columns=['id'])
        fs1_same = FeatureSet(df=df, metadata_columns=['id'])

        # Different metadata_columns
        fs2 = FeatureSet(df=df, metadata_columns=['id', 'x2'])

        # Different value for x2
        df["x2"] = 17
        fs3 = FeatureSet(df=df, metadata_columns=['id'])

        # Check the not equal cases
        self.assertNotEqual(fs1, None)
        self.assertNotEqual(fs1, fs2)
        self.assertNotEqual(fs1, fs3)

        # Check the equal cases
        self.assertEqual(fs1, fs1)
        self.assertEqual(fs1, fs1_same)

    def test_load_save_csv(self):
        # Test that we can load and save CSV files.
        fs = FeatureSet()

        csv_file = "test-feature_set.csv"
        tsv_file = "test-feature_set.tsv"

        # Load the TSV file and save it to a tmp dir.  Make sure the files match, and that at least one field matches
        fs.load_csv(tsv_file, in_dir=test.test_data_dir, sep='\t')
        fs.save_csv(tsv_file, out_dir=test.tmp_data_dir, sep='\t')
        self.assertTrue(filecmp.cmp(os.path.join(test.test_data_dir, tsv_file),
                                    os.path.join(test.tmp_data_dir, tsv_file)), "TSV files did not match.")
        self.assertEqual(fs.get_example_df().loc[0, 'cavity_label'], "5")
        self.assertEqual(fs.get_example_df().loc[0, 'f1'], 6)

        # Clean up the tmp TSV file
        os.unlink(os.path.join(test.tmp_data_dir, tsv_file))

        # Do the same test with CSV file/separator
        fs.load_csv(csv_file, in_dir=test.test_data_dir)
        fs.save_csv(csv_file, out_dir=test.tmp_data_dir)
        self.assertTrue(filecmp.cmp(os.path.join(test.test_data_dir, csv_file),
                                    os.path.join(test.tmp_data_dir, csv_file)), "CSV files did not match.")
        self.assertEqual(fs.get_example_df().loc[0, 'fault_label'], "Microphonics")
        self.assertEqual(fs.get_example_df().loc[2, 'f3'], 3)

        # Clean up the tmp CSV file
        os.unlink(os.path.join(test.tmp_data_dir, csv_file))

    def test_update_example_set(self):
        tsv_file = "test-feature_set.tsv"
        fs = FeatureSet()
        fs.load_csv(tsv_file, in_dir=test.test_data_dir, sep='\t')

        # Add one column and include all of the "mandatory" columns in there.  They should still only show up once.
        m_cols = fs.metadata_columns + ['junk']
        df = fs.get_example_df()
        df['junk'] = ['junker'] * len(df)
        fs.update_example_set(df, metadata_columns=m_cols)

        self.assertListEqual(fs.metadata_columns, m_cols)


if __name__ == '__main__':
    unittest.main()
