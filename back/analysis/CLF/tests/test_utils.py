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
    """测试数据类型推断功能的测试类"""
    
    def setUp(self):
        """创建测试数据"""
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
        """测试空值处理"""
        cleaned_df = clean_special_values(self.df)
        # 检查是否正确处理了空值
        self.assertTrue(pd.isna(cleaned_df.loc[3, 'Name']))
        self.assertTrue(pd.isna(cleaned_df.loc[4, 'Name']))
        self.assertTrue(pd.isna(cleaned_df.loc[2, 'Score']))

    def test_date_conversion(self):
        """测试日期转换"""
        df = infer_and_convert_data_types(self.df)
        # 检查日期格式是否正确转换为澳洲格式
        self.assertEqual(
            df.loc[0, 'Birthdate'].strftime('%d/%m/%Y'),
            '01/01/1990'
        )

    def test_boolean_conversion(self):
        """测试布尔值转换"""
        df = infer_and_convert_data_types(self.df)
        # 检查布尔值是否正确转换
        self.assertEqual(df.loc[0, 'Is_Student'], True)
        self.assertEqual(df.loc[1, 'Is_Student'], False)

    def test_preview_data_generation(self):
        """测试预览数据生成"""
        df = infer_and_convert_data_types(self.df)
        preview = generate_preview_data(df)
        # 检查预览数据的格式和内容
        self.assertEqual(len(preview), min(5, len(df)))
        self.assertTrue(all(isinstance(row, dict) for row in preview))

    def test_sample_value_generation(self):
        """测试样本值生成"""
        df = infer_and_convert_data_types(self.df)
        sample = get_column_sample(df, 'Score')
        # 检查样本值是否正确生成
        self.assertEqual(sample, '90')