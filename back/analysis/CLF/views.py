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

            # file check for csv and excel
            if not file.name.endswith(('.csv', '.xlsx')):
                return Response(
                    {'error': 'Invalid file type. Please upload CSV or Excel file'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            try:
                # read file
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

            # data type inference
            logger.info("Starting data type inference")
            df = optimize_dataframe(df)
            df = infer_and_convert_data_types(df)
            logger.info("Completed data type inference")

            # prepare response data
            types = {}
            samples = {}
            inference_info = {}  #store inference information
            
            for column in df.columns:
                try:
                    # get data type
                    types[column] = str(df[column].dtype)
                    
                    # first 5 non-null values (sample values)
                    samples[column] = get_column_sample(df, column)
                    
                    non_null_values = df[column].dropna().tolist()
                    if len(non_null_values) > 0:
                        # check if column is complex data
                        is_complex = is_complex_data(non_null_values, column)
                        if is_complex:
                            # receive the model inference and confidence
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

            # unique file id for each file
            file_id = generate_file_id()
            save_dataframe(file_id, df)

            response_data = {
                'types': types,
                'samples': samples,
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

            # get the saved DataFrame
            df = get_dataframe(file_id)
            
            print(df)

            # transfer the specific column type
            try:
                # number type
                if new_type == 'number':
                    df[column] = pd.to_numeric(df[column], errors='coerce')

                # datetime type
                elif new_type == 'datetime':
                    # complexe date format
                    if df[column].dtype in ['int64', 'float64']:
                        # ignore seconds and nanoseconds
                        df[column] = pd.to_datetime(df[column], unit='ns')
                    else:
                        # check the date format
                        sample_dates = df[column].dropna().iloc[:5].tolist()
                        date_formats = [
                        '%d/%m/%Y',     # 31/12/2023
                        '%d/%m/%y',     # 31/12/23
                        '%d-%m-%Y',     # 31-12-2023
                        '%d.%m.%Y',     # 31.12.2023
                    ]

                        # the best date format
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
                            # use the best format to convert
                            df[column] = pd.to_datetime(df[column], format=best_format, errors='coerce')
                        else:
                            # use the general format to convert
                            df[column] = pd.to_datetime(df[column], errors='coerce')

                # boolean type
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

                # save the updated DataFrame
                save_dataframe(file_id, df)

                # prepare preview data
                preview_data = []
                for _, row in df.head(5).iterrows():
                    row_dict = {}
                    for col in df.columns:
                        value = row[col]
                        if pd.isna(value):
                            row_dict[col] = "No data available"
                        elif isinstance(value, (np.datetime64, pd.Timestamp)):
                            # australia date format
                            row_dict[col] = value.strftime('%d/%m/%Y')
                        elif isinstance(value, (np.floating, float)):
                            if np.isnan(value):
                                row_dict[col] = "No data available"
                            else:
                                row_dict[col] = f"{float(value):.2f}" if value % 1 != 0 else str(int(value))
                        else:
                            row_dict[col] = str(value)
                    preview_data.append(row_dict)

                    # debug using print
                    # print(preview_data)


                # get the sample value
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
                        sample_value = first_value.strftime('%m/%d/%Y')  # day/month/year
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