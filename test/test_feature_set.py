import unittest

from unittest import TestCase
from rfwtools.feature_set import FeatureSet
import pandas as pd
import numpy as np


def make_dummy_df(n, standard=False):
    """Generates a dummy DataFrame with id as only metadtata column.  Optionally use standard (zone, etc.).

    Value of columns is not same as real data.
    """

    if standard:
        df = pd.DataFrame({'zone': [1] * n,
                           "dtime": [1] * n,
                           'cavity_label': [1] * n,
                           'fault_label': [1] * n,
                           'label_source': [1] * n,
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
        df = make_dummy_df(3)

        # Check that the missing metadata columns throws
        with self.assertRaises(ValueError):
            FeatureSet(df=df)

        # Check that we can supply custom metadata_columns values
        FeatureSet(df=df, metadata_columns=['id'])

        # Construct a more standard looking one.
        df = make_dummy_df(3, standard=True)

        # Test that standard construction works
        FeatureSet(df)
        FeatureSet(df, name="testing!")

    def test_do_pca_reduction(self):
        # Test with a _very_ simple case
        # Create a dummy DataFrame with only one column having variation - should have only on non-zero PC
        df = make_dummy_df(3)
        fs = FeatureSet(df=df, metadata_columns=['id'])

        # Defaults to three components.  Test the no standardization option
        fs.do_pca_reduction(standardize=False, report=False)
        exp = pd.DataFrame({"id": [0, 1, 2], "pc1": [1., 0., -1.], "pc2": [0., 0., 0.], "pc3": [0., 0., 0.]})
        self.assertTrue((exp.equals(fs.get_pca_df())), f"exp = \n{exp}\npca_df = \n{fs.get_pca_df()}")

        # Check that this works with standardization
        fs.do_pca_reduction(standardize=True, report=False)
        exp = pd.DataFrame({"id": [0, 1, 2], "pc1": [1.224744871391589, 0., -1.224744871391589], "pc2": [0., 0., 0.],
                            "pc3": [0., 0., 0.]})
        self.assertTrue((exp.equals(fs.get_pca_df())), f"exp = \n{exp}\npca_df = \n{fs.get_pca_df()}")

        # Check that the explained variance is what we expect (all on one PC)
        self.assertTrue((np.array([1., 0., 0.]) == fs.pca.explained_variance_ratio_).all())

    def test_eq(self):
        # Test our equality operator

        # Make some Feature Sets
        df = make_dummy_df(4)

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


if __name__ == '__main__':
    unittest.main()
