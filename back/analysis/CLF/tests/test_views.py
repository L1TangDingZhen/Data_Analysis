from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
import json
import pandas as pd
import io

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


        # tests/test_views.py 中添加
class StatisticsViewTests(TestCase):
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