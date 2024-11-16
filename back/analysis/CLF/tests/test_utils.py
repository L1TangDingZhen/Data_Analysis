import unittest
import pandas as pd
import numpy as np
from django.test import TestCase
from ..utils import (
    infer_and_convert_data_types, 
    clean_special_values, 
    generate_preview_data,
    get_column_sample
)

class DataTypeInferenceTests(TestCase):
    # test data
    def setUp(self):
        # create test data
        self.test_data = {
            'Name': ['Alice', 'Bob', 'Charlie', None, ''],
            'Grade': ['A', 'B', 'A', 'C', None],
            'Score': ['90', '75', 'N/A', '85', 'not available'],
            'Birthdate': ['1/1/1990', '2/2/1991', '3/3/1992', '20240312', None],
            'Is_Student': ['1', '0', 'true', 'yes', 'n'],
            'Percentage': ['50%', '75.5%', '0.25', None, '100%'],
        }
        self.df = pd.DataFrame(self.test_data)

    def test_null_value_handling(self):
        # clean special values
        cleaned_df = clean_special_values(self.df)
        # check if special values are correctly cleaned
        self.assertTrue(pd.isna(cleaned_df.loc[3, 'Name']))
        self.assertTrue(pd.isna(cleaned_df.loc[4, 'Name']))
        self.assertTrue(pd.isna(cleaned_df.loc[2, 'Score']))

    def test_date_conversion(self):
        # infer and convert data types
        df = infer_and_convert_data_types(self.df)
        # check if dates are correctly converted to the Australian format
        self.assertEqual(
            df.loc[0, 'Birthdate'].strftime('%d/%m/%Y'),
            '01/01/1990'
        )

    def test_boolean_conversion(self):
        # infer and convert data types
        df = infer_and_convert_data_types(self.df)
        self.assertEqual(df.loc[0, 'Is_Student'], True)
        self.assertEqual(df.loc[1, 'Is_Student'], False)

    def test_preview_data_generation(self):
        # infer and convert data types
        df = infer_and_convert_data_types(self.df)
        preview = generate_preview_data(df)
        # check the format and content of the preview data
        self.assertEqual(len(preview), min(5, len(df)))
        self.assertTrue(all(isinstance(row, dict) for row in preview))

    def test_sample_value_generation(self):
        # infer and convert data types
        df = infer_and_convert_data_types(self.df)
        sample = get_column_sample(df, 'Score')
        # check if the sample value is correctly generated
        self.assertEqual(sample, '90')