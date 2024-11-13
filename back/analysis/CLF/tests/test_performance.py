import time
import pandas as pd
import numpy as np
from django.test import TestCase
from ..utils import infer_and_convert_data_types

class PerformanceTests(TestCase):
    """性能测试类"""

    def setUp(self):
        """创建大型测试数据"""
        self.large_df = pd.DataFrame({
            'id': range(10000),
            'name': ['test'] * 10000,
            'value': [1.234567] * 10000,
            'date': pd.date_range('2020-01-01', periods=10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000)
        })

    def test_processing_time(self):
        """测试处理时间"""
        start_time = time.time()
        _ = infer_and_convert_data_types(self.large_df)
        processing_time = time.time() - start_time
        
        # 确保处理时间在合理范围内（例如小于5秒）
        self.assertLess(processing_time, 5)
        print(f"Processing time: {processing_time} seconds")

    def test_memory_usage(self):
        """测试内存使用"""
        initial_memory = self.large_df.memory_usage(deep=True).sum()
        df = infer_and_convert_data_types(self.large_df)
        final_memory = df.memory_usage(deep=True).sum()
        
        # 验证内存使用是否优化
        self.assertLess(final_memory, initial_memory)
        print(f"Memory usage reduced from {initial_memory} to {final_memory} bytes")