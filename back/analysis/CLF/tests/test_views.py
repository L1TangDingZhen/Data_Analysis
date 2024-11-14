from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
import json
import pandas as pd
import io


class BaseTestCase(TestCase):
    # bascic test
    def setUp(self):
        self.client = Client()
        self.setup_test_file()

    def setup_test_file(self):
        # create basic test file
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
    # analysis view test
    def setUp(self):
        # create client
        self.client = Client()
        
        # create test csv file content
        csv_data = """Name,Grade,Score,Birthdate,Is_Student
            Alice,A,90,1/1/1990,1
            Bob,B,75,2/2/1991,0
            Charlie,A,85,3/3/1992,1"""
        
        # create test file
        self.test_file = SimpleUploadedFile(
            "test.csv",
            csv_data.encode('utf-8'),
            content_type='text/csv'
        )

    def test_file_upload(self):
        # file upload test
        # send post request
        response = self.client.post(
            reverse('analyze_file'),
            {'file': self.test_file},
            format='multipart'
        )
        
        # check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # check data structure
        self.assertIn('types', data)
        self.assertIn('samples', data)
        self.assertIn('preview_data', data)
        self.assertIn('file_id', data)

    def test_update_type(self):
        # update type test
        # upload file first
        response = self.client.post(
            reverse('analyze_file'),
            {'file': self.test_file},
            format='multipart'
        )
        file_id = response.json()['file_id']
        
        # test update type
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
        # create test environment
        self.client = Client()
        
        # create test csv file content
        csv_data = """Name,Grade,Score,Birthdate,Is_Student
            Alice,A,90,1/1/1990,1
            Bob,B,75,2/2/1991,0
            Charlie,A,85,3/3/1992,1"""
                    
        # create test file
        self.test_file = SimpleUploadedFile(
            "test.csv",
            csv_data.encode('utf-8'),
            content_type='text/csv'
        )

    def test_export_data(self):
        # upload file first
        response = self.client.post(
            reverse('analyze_file'),
            {'file': self.test_file},
            format='multipart'
        )
        file_id = response.json()['file_id']
        # test export
        export_response = self.client.get(
            reverse('export_data', kwargs={'file_id': file_id})
        )
        self.assertEqual(export_response.status_code, 200)


class StatisticsViewTests(TestCase):
    def setUp(self):
        # create test environment
        self.client = Client()
        
        # create test csv file content
        csv_data = """Name,Grade,Score,Birthdate,Is_Student
            Alice,A,90,1/1/1990,1
            Bob,B,75,2/2/1991,0
            Charlie,A,85,3/3/1992,1"""
        
        # create test file
        self.test_file = SimpleUploadedFile(
            "test.csv",
            csv_data.encode('utf-8'),
            content_type='text/csv'
        )

    def test_statistics_generation(self):
        # upload file
        response = self.client.post(
            reverse('analyze_file'),
            {'file': self.test_file},
            format='multipart'
        )
        file_id = response.json()['file_id']
        
        # test statistics generation
        stats_response = self.client.get(
            reverse('get_statistics', kwargs={'file_id': file_id})
        )
        self.assertEqual(stats_response.status_code, 200)
        data = stats_response.json()
        
        # check data structure
        self.assertIn('numeric_columns', data)
        self.assertIn('categorical_columns', data)
        self.assertIn('datetime_columns', data)



class ComplexDataAnalysisTests(TestCase):
    # complex data analysis test
    def setUp(self):
        self.client = Client()
        
        # fix csv format to ensure consistent number of fields per row
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
        # complex data detection test
        response = self.client.post(
            reverse('analyze_file'),
            {'file': self.test_file},
            format='multipart'
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # check if inference_info exists
        self.assertIn('inference_info', data)
        
        # check if Mixed_Column is detected as complex data
        self.assertTrue(
            data['inference_info']['Mixed_Column']['is_complex']
        )
        
        # check if model was involved in analysis
        self.assertIn('used_model', data['inference_info']['Mixed_Column'])

    def test_spacy_model_integration(self):
        # spacy model integration test
        response = self.client.post(
            reverse('analyze_file'),
            {'file': self.test_file},
            format='multipart'
        )
        
        data = response.json()
        
        # check if Currency column is correctly identified
        self.assertIn('Currency', data['inference_info'])
        if data['inference_info']['Currency']['is_complex']:
            self.assertGreater(
                data['inference_info']['Currency'].get('confidence', 0),
                0.5
            )

class DataTypeConversionTests(TestCase):
    # data type conversion test
    
    def setUp(self):
        self.client = Client()
        
        # create test data
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
        # type inference accuracy test
        response = self.client.post(
            reverse('analyze_file'),
            {'file': self.test_file},
            format='multipart'
        )
        
        data = response.json()
        
        # check basic type inference
        self.assertEqual(data['types']['Text'], 'object')   # should be object type
        self.assertIn(data['types']['Number'], ['int64', 'int32']) # should be integer type
        
        # check mixed data column
        self.assertTrue(data['inference_info']['Mixed']['is_complex'])

    def test_error_handling(self):
        # error handling test
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
        
        self.assertEqual(response.status_code, 200)  # should still process successfully
        data = response.json()
        
        # check error handling
        self.assertIn('Number', data['types'])
        self.assertIn('Date', data['types'])

class DataTypeConversionTests(TestCase):
    def setUp(self):
        self.client = Client()
        
        # fix csv format by removing illegal commas
        csv_data = """Text,Number,Date,Mixed
            Regular text,123,2024-01-01,123.45
            More text,456,2024-01-02,ABC123
            Final text,789,2024-01-03,XYZ789"""
        
        self.test_file = SimpleUploadedFile(
            "conversion_test.csv",
            csv_data.encode('utf-8'),
            content_type='text/csv'
        )