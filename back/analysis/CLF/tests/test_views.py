from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
import json
import pandas as pd
import io


class BaseTestCase(TestCase):
    """基础测试类"""
    def setUp(self):
        self.client = Client()
        self.setup_test_file()

    def setup_test_file(self):
        # 创建基本测试文件
        csv_data = """Name,Grade,Score,Birthdate,Is_Student
Alice,A,90,1/1/1990,1
Bob,B,75,2/2/1991,0
Charlie,A,85,3/3/1992,1"""
        
        self.test_file = SimpleUploadedFile(
            "test.csv",
            csv_data.encode('utf-8'),
            content_type='text/csv'
        )



class FileAnalysisViewTests(TestCase):
    """测试文件分析视图的测试类"""

    def setUp(self):
        """设置测试环境"""
        self.client = Client()
        
        # 创建测试CSV文件内容
        csv_data = """Name,Grade,Score,Birthdate,Is_Student
Alice,A,90,1/1/1990,1
Bob,B,75,2/2/1991,0
Charlie,A,85,3/3/1992,1"""
        
        # 创建测试文件
        self.test_file = SimpleUploadedFile(
            "test.csv",
            csv_data.encode('utf-8'),
            content_type='text/csv'
        )

    def test_file_upload(self):
        """测试文件上传功能"""
        # 发送POST请求
        response = self.client.post(
            reverse('analyze_file'),
            {'file': self.test_file},
            format='multipart'
        )
        
        # 检查响应
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # 验证返回的数据结构
        self.assertIn('types', data)
        self.assertIn('samples', data)
        self.assertIn('preview_data', data)
        self.assertIn('file_id', data)

    def test_update_type(self):
        """测试类型更新功能"""
        # 首先上传文件
        response = self.client.post(
            reverse('analyze_file'),
            {'file': self.test_file},
            format='multipart'
        )
        file_id = response.json()['file_id']
        
        # 测试更新类型
        update_response = self.client.post(
            reverse('update_type'),
            json.dumps({
                'column': 'Grade',
                'new_type': 'category',
                'file_id': file_id
            }),
            content_type='application/json'
        )
        
        self.assertEqual(update_response.status_code, 200)
        data = update_response.json()
        self.assertIn('preview_data', data)
        self.assertIn('new_type', data)


class ExportViewTests(TestCase):
    def setUp(self):
        """设置测试环境"""
        self.client = Client()
        
        # 创建测试CSV文件内容
        csv_data = """Name,Grade,Score,Birthdate,Is_Student
Alice,A,90,1/1/1990,1
Bob,B,75,2/2/1991,0
Charlie,A,85,3/3/1992,1"""
        
        # 创建测试文件
        self.test_file = SimpleUploadedFile(
            "test.csv",
            csv_data.encode('utf-8'),
            content_type='text/csv'
        )

    def test_export_data(self):
        # 首先上传文件
        response = self.client.post(
            reverse('analyze_file'),
            {'file': self.test_file},
            format='multipart'
        )
        file_id = response.json()['file_id']
        
        # 测试导出
        export_response = self.client.get(
            reverse('export_data', kwargs={'file_id': file_id})
        )
        self.assertEqual(export_response.status_code, 200)


class StatisticsViewTests(TestCase):
    def setUp(self):
        """设置测试环境"""
        self.client = Client()
        
        # 创建测试CSV文件内容
        csv_data = """Name,Grade,Score,Birthdate,Is_Student
Alice,A,90,1/1/1990,1
Bob,B,75,2/2/1991,0
Charlie,A,85,3/3/1992,1"""
        
        # 创建测试文件
        self.test_file = SimpleUploadedFile(
            "test.csv",
            csv_data.encode('utf-8'),
            content_type='text/csv'
        )

    def test_statistics_generation(self):
        # 上传文件
        response = self.client.post(
            reverse('analyze_file'),
            {'file': self.test_file},
            format='multipart'
        )
        file_id = response.json()['file_id']
        
        # 测试统计信息生成
        stats_response = self.client.get(
            reverse('get_statistics', kwargs={'file_id': file_id})
        )
        self.assertEqual(stats_response.status_code, 200)
        data = stats_response.json()
        
        # 验证返回的统计信息结构
        self.assertIn('numeric_columns', data)
        self.assertIn('categorical_columns', data)
        self.assertIn('datetime_columns', data)



class ComplexDataAnalysisTests(TestCase):
    """测试复杂数据分析功能"""
    def setUp(self):
        self.client = Client()
        
        # 修复CSV格式，确保每行字段数一致
        csv_data = """Name,Mixed_Column,Date_Column,Currency,Percentage
    Alice,1234.56,2024-01-01,$1000,50%
    Bob,ABC123,2024/01/02,$2000,75%
    Charlie,Test123,2024-01-03,$3000,25%"""
        
        self.test_file = SimpleUploadedFile(
            "complex_test.csv",
            csv_data.encode('utf-8'),
            content_type='text/csv'
        )

    def test_complex_data_detection(self):
        """测试复杂数据检测功能"""
        response = self.client.post(
            reverse('analyze_file'),
            {'file': self.test_file},
            format='multipart'
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # 验证inference_info是否存在
        self.assertIn('inference_info', data)
        
        # 验证Mixed_Column是否被检测为复杂数据
        self.assertTrue(
            data['inference_info']['Mixed_Column']['is_complex']
        )
        
        # 验证模型是否参与了分析
        self.assertIn('used_model', data['inference_info']['Mixed_Column'])

    def test_spacy_model_integration(self):
        """测试spaCy模型集成"""
        response = self.client.post(
            reverse('analyze_file'),
            {'file': self.test_file},
            format='multipart'
        )
        
        data = response.json()
        
        # 验证Currency列是否正确识别
        self.assertIn('Currency', data['inference_info'])
        if data['inference_info']['Currency']['is_complex']:
            self.assertGreater(
                data['inference_info']['Currency'].get('confidence', 0),
                0.5
            )

class DataTypeConversionTests(TestCase):
    """测试数据类型转换功能"""
    
    def setUp(self):
        self.client = Client()
        
        # 创建测试数据
        csv_data = """Text,Number,Date,Mixed
Regular text,123,2024-01-01,123.45
More text,456,Jan 1 2024,$1,234.56
Final text,789,01/01/2024,ABC123"""
        
        self.test_file = SimpleUploadedFile(
            "conversion_test.csv",
            csv_data.encode('utf-8'),
            content_type='text/csv'
        )

    def test_type_inference_accuracy(self):
        """测试类型推断准确性"""
        response = self.client.post(
            reverse('analyze_file'),
            {'file': self.test_file},
            format='multipart'
        )
        
        data = response.json()
        
        # 验证基本类型推断
        self.assertEqual(data['types']['Text'], 'object')  # 应该是文本类型
        self.assertIn(data['types']['Number'], ['int64', 'int32'])  # 应该是整数类型
        
        # 验证混合数据列
        self.assertTrue(data['inference_info']['Mixed']['is_complex'])

    def test_error_handling(self):
        """测试错误处理"""
        # 创建包含错误数据的CSV
        csv_data = """Number,Date
invalid,2024-01-01
123,invalid_date
456,2024-01-01"""
        
        error_file = SimpleUploadedFile(
            "error_test.csv",
            csv_data.encode('utf-8'),
            content_type='text/csv'
        )
        
        response = self.client.post(
            reverse('analyze_file'),
            {'file': error_file},
            format='multipart'
        )
        
        self.assertEqual(response.status_code, 200)  # 应该仍然成功处理
        data = response.json()
        
        # 验证错误处理
        self.assertIn('Number', data['types'])
        self.assertIn('Date', data['types'])

class DataTypeConversionTests(TestCase):
    def setUp(self):
        self.client = Client()
        
        # 修复CSV格式，移除不合法的逗号
        csv_data = """Text,Number,Date,Mixed
Regular text,123,2024-01-01,123.45
More text,456,2024-01-02,ABC123
Final text,789,2024-01-03,XYZ789"""
        
        self.test_file = SimpleUploadedFile(
            "conversion_test.csv",
            csv_data.encode('utf-8'),
            content_type='text/csv'
        )