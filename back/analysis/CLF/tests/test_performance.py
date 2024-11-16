import time
import pandas as pd
import numpy as np
from django.test import TestCase
from ..utils import infer_and_convert_data_types

class PerformanceTests(TestCase):
    # oerformace test

    def setUp(self):
        # create large data
        self.large_df = pd.DataFrame({
            'id': range(10000),
            'name': ['test'] * 10000,
            'value': [1.234567] * 10000,
            'date': pd.date_range('2020-01-01', periods=10000),
            'category': np.random.choice(['A', 'B', 'C'], 10000)
        })

    def test_processing_time(self):
        # test the processing time
        start_time = time.time()
        _ = infer_and_convert_data_types(self.large_df)
        processing_time = time.time() - start_time
        
        # 确保处理时间在合理范围内（例如小于5秒）
        self.assertLess(processing_time, 5)
        print(f"Processing time: {processing_time} seconds")

    def test_memory_usage(self):
        # memory usage before and after optimization
        initial_memory = self.large_df.memory_usage(deep=True).sum()
        df = infer_and_convert_data_types(self.large_df)
        final_memory = df.memory_usage(deep=True).sum()
        
        # verify that the memory usage is optimized
        self.assertLess(final_memory, initial_memory)
        print(f"Memory usage reduced from {initial_memory} to {final_memory} bytes")