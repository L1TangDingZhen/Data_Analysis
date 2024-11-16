from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, JSONParser
import pandas as pd
import numpy as np
# import json
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
    SpacyModelCache
)
from django.http import FileResponse
import io
import logging

logger = logging.getLogger(__name__)

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

            if not file.name.endswith(('.csv', '.xlsx')):
                return Response(
                    {'error': 'Invalid file type. Please upload CSV or Excel file'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            try:
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

            # store inference information
            inference_info = {}
            types = {}
            
            # data type inference
            logger.info("Starting data type inference")
            df = optimize_dataframe(df)


            for column in df.columns:
                try:
                    non_null_values = df[column].dropna().tolist()
                    if len(non_null_values) > 0:
                        is_complex = is_complex_data(non_null_values, column)
                        
                        # check for specific column name patterns
                        col_lower = column.lower()
                        if 'grade' in col_lower or 'level' in col_lower:
                            inferred_type = 'category'
                            confidence = 0.9
                        elif 'country' in col_lower or 'state' in col_lower:
                            inferred_type = 'category'
                            confidence = 0.9
                        elif 'height' in col_lower or 'weight' in col_lower:
                            inferred_type = 'number'
                            confidence = 0.9
                        elif is_complex:
                            inferred_type, confidence = SpacyModelCache.analyze_complex_data(
                                non_null_values, column
                            )
                        else:
                            # basic type inference
                            if all(isinstance(v, bool) or str(v).lower() in ['true', 'false', '1', '0'] 
                                for v in non_null_values):
                                inferred_type = 'boolean'
                                confidence = 1.0
                            elif all(isinstance(v, (int, float)) for v in non_null_values):
                                inferred_type = 'number'
                                confidence = 1.0
                            else:
                                inferred_type = 'text'
                                confidence = 0.0
                                
                        inference_info[column] = {
                            'is_complex': is_complex,
                            'model_inference': inferred_type,
                            'confidence': float(confidence),
                            'used_model': confidence > 0.5
                        }
                        
                        # apply type conversion if confidence is high enough
                        if confidence > 0.5:
                            if inferred_type == 'category':
                                df[column] = pd.Categorical(df[column])
                            elif inferred_type == 'number':
                                df[column] = pd.to_numeric(df[column], errors='coerce')
                            elif inferred_type == 'boolean':
                                bool_map = {'1': True, '0': False, 'true': True, 'false': False}
                                df[column] = df[column].astype(str).str.lower().map(bool_map)
                                
                except Exception as e:
                    logger.error(f"Error processing column {column}: {str(e)}")
                    inference_info[column] = {
                        'is_complex': False,
                        'error': str(e)
                    }
                    
                types[column] = str(df[column].dtype)

            # generate file ID and other response data
            file_id = generate_file_id()
            save_dataframe(file_id, df)

            response_data = {
                'types': types,
                'samples': {col: get_column_sample(df, col) for col in df.columns},
                'rows': len(df),
                'columns': len(df.columns),
                'preview_data': generate_preview_data(df),
                'file_id': file_id,
                'inference_info': inference_info
            }

            logger.info("Successfully prepared response data")
            return Response(response_data)

        except Exception as e:
            logger.error(f"Error in file analysis: {str(e)}")
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

# user can update the data type of a column
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

            # get the dataframe
            df = get_dataframe(file_id)
            original_type = str(df[column].dtype)
            
            logger.info(f"Updating column {column} type from {original_type} to {new_type}")
            
            # convert type only for the specified column
            try:
                if new_type == 'category':
                    df[column] = pd.Categorical(df[column])
                elif new_type == 'number':
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                elif new_type == 'datetime':
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                elif new_type == 'boolean':
                    bool_map = {
                        '1': True, '0': False,
                        'true': True, 'false': False,
                        'yes': True, 'no': False,
                        1: True, 0: False
                    }
                    df[column] = df[column].astype(str).str.lower().map(bool_map)
                else:
                    df[column] = df[column].astype(str)
                
                # save the updated dataframe
                save_dataframe(file_id, df)
                
                logger.info(f"Successfully converted {column} to {new_type}")
                
                return Response({
                    'preview_data': generate_preview_data(df),
                    'inferred_type': str(df[column].dtype),
                    'display_type': new_type,
                    'sample_value': get_column_sample(df, column),
                    'new_type': new_type,  
                    'message': f'Successfully updated type of {column} to {new_type}'
                })

            except Exception as e:
                logger.error(f"Error converting {column} to {new_type}: {str(e)}")
                return Response(
                    {'error': f'Failed to convert {column} to {new_type}: {str(e)}'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

        except Exception as e:
            logger.error(f"Error in update type: {str(e)}")
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ExportDataView(APIView):
    def get(self, request, file_id):
        try:
            logger.info(f"Starting data export for file_id: {file_id}")
            df = get_dataframe(file_id)
            
            # create a byte buffer
            buffer = io.BytesIO()
            # dataframe in utr-8
            df.to_csv(buffer, index=False, encoding='utf-8')
            # move the buffer pointer to the start
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
        

# # future statistics
# class StatisticsView(APIView):
#     # generate statistics
#     def get(self, request, file_id):
#         try:
#             logger.info(f"Generating statistics for file_id: {file_id}")
#             df = get_dataframe(file_id)
            
#             # error handling when generating statistics
#             stats = {
#                 'numeric_columns': {},
#                 'categorical_columns': {},
#                 'datetime_columns': {}
#             }
            
#             try:
#                 # numeric column statistics
#                 numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
#                 for col in numeric_cols:
#                     stats['numeric_columns'][col] = {
#                         'mean': float(df[col].mean()),  # json 
#                         'median': float(df[col].median()),
#                         'std': float(df[col].std()),
#                         'min': float(df[col].min()),
#                         'max': float(df[col].max())
#                     }
                
#                 # categorical column statistics
#                 cat_cols = df.select_dtypes(include=['category', 'object']).columns
#                 for col in cat_cols:
#                     value_counts = df[col].value_counts()
#                     stats['categorical_columns'][col] = {
#                         'value_counts': {str(k): int(v) for k, v in value_counts.items()}, 
#                         'unique_count': int(df[col].nunique())
#                     }
                
#                 # date column statistics
#                 date_cols = df.select_dtypes(include=['datetime64']).columns
#                 for col in date_cols:
#                     stats['datetime_columns'][col] = {
#                         'min_date': df[col].min().strftime('%d/%m/%Y'),
#                         'max_date': df[col].max().strftime('%d/%m/%Y'),
#                         'date_range': int((df[col].max() - df[col].min()).days)
#                     }
                
#                 logger.info("Successfully generated statistics")
#                 return Response(stats)
                
#             except Exception as e:
#                 logger.error(f"Error calculating statistics: {str(e)}")
#                 return Response(
#                     {'error': f'Error calculating statistics: {str(e)}'},
#                     status=status.HTTP_400_BAD_REQUEST
#                 )
                
#         except Exception as e:
#             logger.error(f"Error in statistics generation: {str(e)}")
#             return Response(
#                 {'error': str(e)},
#                 status=status.HTTP_400_BAD_REQUEST
#             )