# views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, JSONParser
import pandas as pd
import json
from .utils import infer_and_convert_data_types
import io
class AnalyzeFileView(APIView):
    parser_classes = (MultiPartParser,)

    def post(self, request):
        try:
            file = request.FILES.get('file')
            if not file:
                return Response(
                    {'error': 'No file provided'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 读取文件
            if file.name.endswith('.csv'):
                df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
            else:
                df = pd.read_excel(file)

            # 推断类型
            df = infer_and_convert_data_types(df)

            # 准备响应数据
            types = {}
            samples = {}
            for column in df.columns:
                types[column] = str(df[column].dtype)
                samples[column] = str(df[column].iloc[0]) if len(df) > 0 else None

            # 获取预览数据（前5行）
            preview_data = df.head(5).to_dict('records')

            # 生成唯一的文件ID并保存DataFrame（你需要实现保存逻辑）
            file_id = generate_file_id()  # 实现这个函数
            save_dataframe(file_id, df)   # 实现这个函数

            return Response({
                'types': types,
                'samples': samples,
                'rows': len(df),
                'columns': len(df.columns),
                'preview_data': preview_data,
                'file_id': file_id
            })

        except Exception as e:
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
            df = get_dataframe(file_id)  # 实现这个函数

            # 转换列类型
            if new_type == 'number':
                df[column] = pd.to_numeric(df[column], errors='coerce')
            elif new_type == 'datetime':
                df[column] = pd.to_datetime(df[column], errors='coerce')
            elif new_type == 'boolean':
                df[column] = df[column].map({'True': True, 'False': False})
            elif new_type == 'category':
                df[column] = pd.Categorical(df[column])
            else:
                df[column] = df[column].astype(str)

            # 更新保存的DataFrame
            save_dataframe(file_id, df)

            # 返回更新后的预览数据
            preview_data = df.head(5).to_dict('records')

            return Response({
                'preview_data': preview_data,
                'message': f'Successfully updated type of {column} to {new_type}'
            })

        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )