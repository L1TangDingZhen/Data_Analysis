from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, JSONParser
import pandas as pd
import numpy as np
import json
import io
from .utils import (
    infer_and_convert_data_types,
    generate_file_id,
    save_dataframe,
    get_dataframe,
    generate_preview_data,
    get_column_sample,
    optimize_dataframe,
    is_complex_data,
    SpacyModelCache  # 添加新的导入
)
from django.http import FileResponse
import io
import logging

logger = logging.getLogger(__name__)



# views.py
class AnalyzeFileView(APIView):
    parser_classes = (MultiPartParser,)

    def post(self, request):
        try:
            logger.info("Received file analysis request")
            file = request.FILES.get('file')
            if not file:
                return Response(
                    {'error': 'No file provided'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 检查文件类型
            if not file.name.endswith(('.csv', '.xlsx')):
                return Response(
                    {'error': 'Invalid file type. Please upload CSV or Excel file'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            try:
                # 读取文件
                if file.name.endswith('.csv'):
                    df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
                else:
                    df = pd.read_excel(file)
                    
                logger.info(f"Successfully read file with shape {df.shape}")
            except Exception as e:
                logger.error(f"Error reading file: {str(e)}")
                return Response(
                    {'error': f'Error reading file: {str(e)}'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 应用类型推断（现在包含了模型分析）
            logger.info("Starting data type inference")
            df = optimize_dataframe(df)
            df = infer_and_convert_data_types(df)
            logger.info("Completed data type inference")

            # 准备响应数据
            types = {}
            samples = {}
            inference_info = {}  # 新增：存储推断信息
            
            for column in df.columns:
                try:
                    # 获取数据类型
                    types[column] = str(df[column].dtype)
                    
                    # 获取样本值
                    samples[column] = get_column_sample(df, column)
                    
                    # 获取非空值用于复杂数据分析
                    non_null_values = df[column].dropna().tolist()
                    if len(non_null_values) > 0:
                        # 检查是否为复杂数据
                        is_complex = is_complex_data(non_null_values, column)
                        if is_complex:
                            # 获取模型分析结果
                            inferred_type, confidence = SpacyModelCache.analyze_complex_data(
                                non_null_values, column
                            )
                            inference_info[column] = {
                                'is_complex': True,
                                'model_inference': inferred_type,
                                'confidence': float(confidence),
                                'used_model': confidence > 0.5
                            }
                        else:
                            inference_info[column] = {
                                'is_complex': False,
                                'used_model': False
                            }
                            
                except Exception as e:
                    logger.error(f"Error processing column {column}: {str(e)}")
                    types[column] = 'object'
                    samples[column] = "Error in processing"
                    inference_info[column] = {
                        'is_complex': False,
                        'error': str(e)
                    }

            # 生成文件ID并保存DataFrame
            file_id = generate_file_id()
            save_dataframe(file_id, df)

            response_data = {
                'types': types,
                'samples': samples,
                'rows': len(df),
                'columns': len(df.columns),
                'preview_data': generate_preview_data(df),
                'file_id': file_id,
                'inference_info': inference_info  # 新增：包含推断信息
            }

            logger.info("Successfully prepared response data")
            return Response(response_data)

        except Exception as e:
            logger.error(f"Error in file analysis: {str(e)}")
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        

class UpdateTypeView(APIView):
    parser_classes = (JSONParser,)

    def post(self, request):
        try:
            column = request.data.get('column')
            new_type = request.data.get('new_type')
            file_id = request.data.get('file_id')

            if not all([column, new_type, file_id]):
                return Response(
                    {'error': 'Missing required fields'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 获取保存的DataFrame
            df = get_dataframe(file_id)
            
            print(df)

            # 转换列类型
            try:
                if new_type == 'number':
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                elif new_type == 'datetime':
                    # 如果当前是时间戳格式（大数字）
                    if df[column].dtype in ['int64', 'float64']:
                        # 将纳秒时间戳转换为日期时间
                        df[column] = pd.to_datetime(df[column], unit='ns')
                    else:
                        # 尝试检测日期格式
                        sample_dates = df[column].dropna().iloc[:5].tolist()
                        date_formats = [
                        '%d/%m/%Y',     # 31/12/2023 (澳洲格式)
                        '%d/%m/%y',     # 31/12/23
                        '%d-%m-%Y',     # 31-12-2023
                        '%d.%m.%Y',     # 31.12.2023
                    ]

                        # 检测最合适的日期格式
                        best_format = None
                        max_success = 0
                        
                        for date_format in date_formats:
                            try:
                                success_count = 0
                                for date_str in sample_dates:
                                    try:
                                        if isinstance(date_str, str):
                                            pd.to_datetime(date_str, format=date_format)
                                            success_count += 1
                                    except:
                                        continue
                                if success_count > max_success:
                                    max_success = success_count
                                    best_format = date_format
                            except:
                                continue

                        if best_format:
                            # 使用检测到的最佳格式转换
                            df[column] = pd.to_datetime(df[column], format=best_format, errors='coerce')
                        else:
                            # 如果没有找到合适的格式，使用通用解析
                            df[column] = pd.to_datetime(df[column], errors='coerce')

                elif new_type == 'boolean':
                    bool_map = {
                        'True': True, 'False': False,
                        'true': True, 'false': False,
                        'TRUE': True, 'FALSE': False,
                        'T': True, 'F': False,
                        't': True, 'f': False,
                        'Yes': True, 'No': False,
                        'yes': True, 'no': False,
                        'Y': True, 'N': False,
                        'y': True, 'n': False,
                        '1': True, '0': False,
                        1: True, 0: False
                    }
                    df[column] = df[column].map(bool_map)
                elif new_type == 'category':
                    df[column] = pd.Categorical(df[column])
                else:  # text
                    df[column] = df[column].astype(str)

                # 更新保存的DataFrame
                save_dataframe(file_id, df)

                # 准备预览数据
                preview_data = []
                for _, row in df.head(5).iterrows():
                    row_dict = {}
                    for col in df.columns:
                        value = row[col]
                        if pd.isna(value):
                            row_dict[col] = "No data available"
                        elif isinstance(value, (np.datetime64, pd.Timestamp)):
                            # 使用澳洲日期格式
                            row_dict[col] = value.strftime('%d/%m/%Y')
                        elif isinstance(value, (np.floating, float)):
                            if np.isnan(value):
                                row_dict[col] = "No data available"
                            else:
                                row_dict[col] = f"{float(value):.2f}" if value % 1 != 0 else str(int(value))
                        else:
                            row_dict[col] = str(value)
                    preview_data.append(row_dict)

                    print(preview_data)


                # 准备示例值
                sample_value = None
                non_null_values = df[column].dropna()
                if not non_null_values.empty:
                    first_value = non_null_values.iloc[0]
                    if pd.isna(first_value):
                        sample_value = "No data available"
                    elif isinstance(first_value, (np.floating, float)):
                        sample_value = f"{float(first_value):.2f}" if first_value % 1 != 0 else str(int(first_value))
                    elif isinstance(first_value, (np.integer, int)):
                        sample_value = str(int(first_value))
                    elif isinstance(first_value, (np.datetime64, pd.Timestamp)):
                        sample_value = first_value.strftime('%m/%d/%Y')  # 使用固定的输出格式
                    else:
                        sample_value = str(first_value).strip()
                else:
                    sample_value = "No data available"

                return Response({
                    'preview_data': preview_data,
                    'new_type': str(df[column].dtype),
                    'sample_value': sample_value,
                    'message': f'Successfully updated type of {column} to {new_type}'
                })

            except Exception as e:
                print(f"Error in type conversion: {str(e)}")
                return Response(
                    {'error': f'Failed to convert type: {str(e)}'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

        except Exception as e:
            print(f"Error in update type: {str(e)}")
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        





class ExportDataView(APIView):
    def get(self, request, file_id):
        try:
            logger.info(f"Starting data export for file_id: {file_id}")
            df = get_dataframe(file_id)
            
            # 创建一个字节缓冲区
            buffer = io.BytesIO()
            # 将DataFrame写入缓冲区，编码为utf-8
            df.to_csv(buffer, index=False, encoding='utf-8')
            # 将缓冲区指针移到开始
            buffer.seek(0)
            
            logger.info("Successfully exported data")
            response = FileResponse(
                buffer,
                as_attachment=True,
                filename='processed_data.csv'
            )
            return response
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        
class StatisticsView(APIView):
    """生成数据统计信息"""
    def get(self, request, file_id):
        try:
            logger.info(f"Generating statistics for file_id: {file_id}")
            df = get_dataframe(file_id)
            
            # 生成统计信息时添加错误处理
            stats = {
                'numeric_columns': {},
                'categorical_columns': {},
                'datetime_columns': {}
            }
            
            try:
                # 数值列统计
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                for col in numeric_cols:
                    stats['numeric_columns'][col] = {
                        'mean': float(df[col].mean()),  # 确保可JSON序列化
                        'median': float(df[col].median()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max())
                    }
                
                # 分类列统计
                cat_cols = df.select_dtypes(include=['category', 'object']).columns
                for col in cat_cols:
                    value_counts = df[col].value_counts()
                    stats['categorical_columns'][col] = {
                        'value_counts': {str(k): int(v) for k, v in value_counts.items()},  # 确保键和值都是可序列化的
                        'unique_count': int(df[col].nunique())
                    }
                
                # 日期列统计
                date_cols = df.select_dtypes(include=['datetime64']).columns
                for col in date_cols:
                    stats['datetime_columns'][col] = {
                        'min_date': df[col].min().strftime('%d/%m/%Y'),
                        'max_date': df[col].max().strftime('%d/%m/%Y'),
                        'date_range': int((df[col].max() - df[col].min()).days)
                    }
                
                logger.info("Successfully generated statistics")
                return Response(stats)
                
            except Exception as e:
                logger.error(f"Error calculating statistics: {str(e)}")
                return Response(
                    {'error': f'Error calculating statistics: {str(e)}'},
                    status=status.HTTP_400_BAD_REQUEST
                )
                
        except Exception as e:
            logger.error(f"Error in statistics generation: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )