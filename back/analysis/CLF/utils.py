import pandas as pd
import uuid
from django.core.cache import cache
import pickle
import numpy as np
import re
import logging
from typing import Dict, List, Any
import spacy
from typing import Optional, Dict, Any, List, Tuple


# 旧的logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)



# def is_complex_data(values: List[Any], column_name: str) -> bool:
#     """
#     基于数据本身的复杂性判断，而不是预设的列名
#     复杂数据的判断标准：
#     1. 数据表示方式不一致（如同一概念有多种写法）
#     2. 数据类型混杂
#     3. 格式不统一
#     """
#     try:
#         # 获取非空值
#         clean_values = [v for v in values if pd.notna(v)]
#         if not clean_values:
#             return False

#         # 数据特征收集
#         characteristics = {
#             'formats': set(),      # 数据格式
#             'types': set(),        # 数据类型
#             'representations': set()  # 数据表示方式
#         }

#         # 分析每个值的特征
#         for value in clean_values:
#             orig_value = value
#             str_value = str(value).strip().lower()
            
#             # 1. 检查原始数据类型
#             if isinstance(orig_value, (int, float)):
#                 characteristics['types'].add('number')
#             elif isinstance(orig_value, pd.Timestamp):
#                 characteristics['types'].add('datetime')
#             else:
#                 # 2. 尝试类型转换
#                 try:
#                     float(str_value)
#                     characteristics['types'].add('number')
#                 except ValueError:
#                     try:
#                         pd.to_datetime(str_value)
#                         characteristics['types'].add('datetime')
#                     except:
#                         characteristics['types'].add('text')

#             # 3. 检查数据格式特征
#             if isinstance(orig_value, str):
#                 # 分隔符检查
#                 if '/' in str_value or '-' in str_value or '.' in str_value:
#                     characteristics['formats'].add('separated')
#                 # 大小写变化检查
#                 if str_value != str(orig_value).strip():
#                     characteristics['formats'].add('case_variant')
#                 # 长度检查（区分缩写和全称）
#                 characteristics['formats'].add(f'len_{len(str_value)}')
#                 # 数字和文本混合检查
#                 if any(c.isdigit() for c in str_value) and any(c.isalpha() for c in str_value):
#                     characteristics['formats'].add('mixed')

#             # 4. 记录不同的表示方式
#             characteristics['representations'].add(str_value)

#         # 评估复杂性
#         is_complex = (
#             # 类型混杂
#             len(characteristics['types']) > 1 or
#             # 格式不统一（包括分隔符、大小写、长度等）
#             len(characteristics['formats']) > 2 or
#             # 同一概念有多种表示方式
#             (len(characteristics['representations']) > len(set(str(v).strip().lower() for v in clean_values)) * 0.5) or
#             # 存在缩写和全称混用
#             (any(f'len_1' in f or f'len_2' in f for f in characteristics['formats']) and
#              any(f'len_{i}' in f for f in characteristics['formats'] for i in range(3, 10)))
#         )

#         logger.info(f"Column {column_name} complexity check: {is_complex}")
#         if is_complex:
#             logger.info(f"Column characteristics: {characteristics}")

#         return is_complex

#     except Exception as e:
#         logger.warning(f"Error checking complexity for {column_name}: {str(e)}")
#         return False

def is_complex_data(values: List[Any], column_name: str) -> bool:
    # 修改评估复杂性的逻辑
    try:
        clean_values = [v for v in values if pd.notna(v)]
        if not clean_values:
            return False
            
        characteristics = {
            'formats': set(),
            'types': set(),
            'representations': set()
        }
        
        # 基础类型检查
        if all(isinstance(v, (int, float)) for v in clean_values):
            return False
        if all(isinstance(v, bool) for v in clean_values):
            return False
        
        # 数据特征分析
        for value in clean_values:
            str_value = str(value).strip().lower()
            
            # 类型检查
            if isinstance(value, (int, float)):
                characteristics['types'].add('number')
            elif isinstance(value, pd.Timestamp):
                characteristics['types'].add('datetime') 
            else:
                try:
                    float(str_value)
                    characteristics['types'].add('number')
                except ValueError:
                    try:
                        pd.to_datetime(str_value)
                        characteristics['types'].add('datetime')
                    except:
                        characteristics['types'].add('text')
            
            # 数据表示方式检查        
            characteristics['representations'].add(str_value)
            
        # 判断复杂性条件
        is_complex = (
            len(characteristics['types']) > 1 or  # 多种数据类型混合
            (len(characteristics['representations']) / len(clean_values) > 0.8)  # 数据表示方式过于分散
        )
        
        return is_complex
        
    except Exception as e:
        logger.warning(f"Error checking complexity for {column_name}: {str(e)}")
        return False









#旧的
# class SpacyModelCache:
#     _instance = None
#     _model = None

#     @classmethod
#     def get_model(cls) -> spacy.language.Language:
#         # load spacy model
#         if cls._model is None:
#             try:
#                 cls._model = spacy.load('en_core_web_md')
#                 logger.info("Successfully loaded spaCy model")
#             except Exception as e:
#                 logger.error(f"Error loading spaCy model: {str(e)}")
#                 raise
#         return cls._model

#     @classmethod
#     def analyze_complex_data(cls, 
#             values: List[Any], 
#             column_name: str) -> Tuple[str, float]:
        
#         # complex data analysis
#         # values is a list of values in the column
#         # column_name is the name of the column
#         # return the inferred type and confidence (type and confidence)
#         try:
#             model = cls.get_model()
#             predictions = []
#             confidence_scores = []
            
#             # 10 non-null values for analysis
#             sample_values = [str(v) for v in values[:10] if pd.notna(v)]
            
#             for value in sample_values:
#                 doc = model(value)
                
#                 # get entity type
#                 if doc.ents:
#                     ent = doc.ents[0]
#                     predictions.append(cls._map_entity_to_type(ent.label_))
#                     confidence_scores.append(ent._.trf_score if hasattr(ent._, 'trf_score') else 0.5)
#                 else:
#                     # get type from POS
#                     pos_type = cls._get_type_from_pos(doc)
#                     if pos_type:
#                         predictions.append(pos_type)
#                         confidence_scores.append(0.3)  # low confidence
            
#             if predictions:
#                 # most common prediction
#                 most_common = max(set(predictions), key=predictions.count)
#                 confidence = sum(c for p, c in zip(predictions, confidence_scores) 
#                     if p == most_common) / len(predictions)
#                 return most_common, confidence
            
#             return 'text', 0.0  # return text type if no entities found
            
#         except Exception as e:
#             logger.warning(f"Error in complex data analysis: {str(e)}")
#             return 'text', 0.0
            
#     @staticmethod
#     def _map_entity_to_type(ent_label: str) -> str:
#         # entity label to data type mapping
#         type_mapping = {
#             'DATE': 'datetime',
#             'TIME': 'datetime',
#             'CARDINAL': 'number',
#             'MONEY': 'number',
#             'PERCENT': 'number',
#             'QUANTITY': 'number',
#             'ORDINAL': 'number'
#         }
#         return type_mapping.get(ent_label, 'text')
        
#     @staticmethod
#     def _get_type_from_pos(doc: spacy.tokens.Doc) -> Optional[str]:
#         # predict type based on POS
#         # main POS tag
#         main_pos = doc[0].pos_
        
#         if main_pos in ['NUM']:
#             return 'number'
#         elif main_pos in ['PROPN', 'NOUN']:
#             return 'category'
            
#         return None


# # complex data detection
# def is_complex_data(values: List[Any], column_name: str) -> bool:
#     # check if the column contains complex data
#     # return True if complex data is detected, False otherwise
#     try:
#         # non null values
#         clean_values = [v for v in values if pd.notna(v)]
#         if not clean_values:
#             return False
            
#         # first 10 values
#         value_types = set()
#         for value in clean_values[:10]:  # check first 10 values
#             # if it's a pandas Timestamp
#             if isinstance(value, pd.Timestamp):
#                 value_types.add('date')
#             # if it's a number
#             elif isinstance(value, (int, float)):
#                 value_types.add('number')
#             elif isinstance(value, str):
#                 # if it's a string, check if it can be converted to other types
#                 try:
#                     float(value)
#                     value_types.add('number')
#                     continue
#                 except ValueError:
#                     pass
                
#                 # convert to datetime
#                 try:
#                     pd.to_datetime(value)
#                     value_types.add('date')
#                     continue
#                 except:
#                     pass
                
#                 # check if it's a currency format (e.g., $1,234.56)
#                 if re.match(r'^\$?\d{1,3}(,\d{3})*(\.\d+)?$', value):
#                     value_types.add('currency')
#                     continue
                    
#                 # e.g., 10%, 10.5%, 0.5%
#                 if re.match(r'^\d+(\.\d+)?%$', value):
#                     value_types.add('percentage')
#                     continue
                
#                 # other cases are considered as text
#                 value_types.add('text')
        
#         # multiple types or special types (currency, percentage) are considered complex
#         special_types = {'currency', 'percentage'}
#         return len(value_types) > 1 or bool(value_types.intersection(special_types))
            
#     except Exception as e:
#         logger.warning(f"Error checking complex data: {str(e)}")
#         return False







#新的
# class SpacyModelCache:
#     _instance = None
#     _model = None

#     @classmethod
#     def get_model(cls):
#         if cls._model is None:
#             try:
#                 cls._model = spacy.load('en_core_web_md')
#                 logger.info("Successfully loaded spaCy model")
#             except Exception as e:
#                 logger.error(f"Error loading spaCy model: {str(e)}")
#                 raise
#         return cls._model




class SpacyModelCache:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = spacy.load('en_core_web_md')
            logger.info("Successfully loaded spaCy model")
        return cls._model

    @classmethod
    def analyze_complex_data(cls, values: List[Any], column_name: str) -> Tuple[str, float]:
        """分析复杂数据，包含详细日志"""
        try:
            logger.info(f"Starting SpaCy analysis for column: {column_name}")
            
            # 检查日期格式
            date_matches = sum(1 for v in values if cls._is_date_format(str(v)))
            date_confidence = date_matches / len(values)
            logger.info(f"Date format check for {column_name}: {date_confidence:.2f}")
            if date_confidence > 0.8:
                logger.info(f"Detected datetime type for {column_name}")
                return 'datetime', date_confidence

            # 检查布尔值
            bool_matches = sum(1 for v in values if str(v).lower() in {'0', '1', 'true', 'false', 'yes', 'no'})
            bool_confidence = bool_matches / len(values)
            logger.info(f"Boolean check for {column_name}: {bool_confidence:.2f}")
            if bool_confidence > 0.8:
                logger.info(f"Detected boolean type for {column_name}")
                return 'boolean', bool_confidence

            # 检查性别数据
            gender_terms = {'m', 'f', 'male', 'female', 'boy', 'girl'}
            gender_matches = sum(1 for v in values if str(v).lower() in gender_terms)
            gender_confidence = gender_matches / len(values)
            logger.info(f"Gender terms check for {column_name}: {gender_confidence:.2f}")
            if gender_confidence > 0.8:
                logger.info(f"Detected category type (gender) for {column_name}")
                return 'category', gender_confidence

            # 检查数值
            numeric_matches = sum(1 for v in values if cls._is_numeric(v))
            numeric_confidence = numeric_matches / len(values)
            logger.info(f"Numeric check for {column_name}: {numeric_confidence:.2f}")
            if numeric_confidence > 0.8:
                logger.info(f"Detected number type for {column_name}")
                return 'number', numeric_confidence

            # 检查分类数据
            unique_ratio = len(set(str(v).lower() for v in values)) / len(values)
            category_confidence = 0.9 if unique_ratio < 0.5 else 0.0
            logger.info(f"Category check for {column_name}: {category_confidence:.2f}")
            if category_confidence > 0.8:
                logger.info(f"Detected category type for {column_name}")
                return 'category', category_confidence

            logger.info(f"No specific type detected for {column_name}, defaulting to text")
            return 'text', 0.0

        except Exception as e:
            logger.error(f"Error in SpaCy analysis for {column_name}: {str(e)}")
            return 'text', 0.0

    @staticmethod
    def _is_date_format(value: str) -> bool:
        date_patterns = [
            r'^\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}$',
            r'^\d{4}[-/\.]\d{1,2}[-/\.]\d{2}$'
        ]
        return any(re.match(p, value) for p in date_patterns)

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        try:
            float(str(value))
            return True
        except:
            return False




#old
# def is_complex_data(values: List[Any], column_name: str) -> bool:
#     """
#     Determine if a column contains complex data that requires advanced processing
#     """
#     try:
#         # Get clean non-null values for analysis
#         clean_values = [v for v in values if pd.notna(v)][:10]
#         if not clean_values:
#             return False

#         # Track type variations
#         value_types = set()
#         format_variations = 0

#         for value in clean_values:
#             if isinstance(value, pd.Timestamp):
#                 value_types.add('datetime')
#             elif isinstance(value, (int, float)):
#                 value_types.add('number')
#             elif isinstance(value, str):
#                 # Check currency format
#                 if re.match(r'^[$£€¥]\s*\d+[.,]?\d*\s*[kKmMbB]?$', value):
#                     value_types.add('currency')
#                     continue

#                 # Check percentage format
#                 if re.match(r'^\d+(\.\d+)?%$', value):
#                     value_types.add('percentage')
#                     continue

#                 # Check mixed formats (e.g., "100 units", "50 kg")
#                 if re.match(r'^\d+(\.\d+)?\s*[a-zA-Z]+$', value):
#                     format_variations += 1
#                     continue

#                 # Try numeric conversion
#                 try:
#                     float(value)
#                     value_types.add('number')
#                 except ValueError:
#                     pass

#                 # Check for dates with various formats
#                 try:
#                     pd.to_datetime(value)
#                     value_types.add('datetime')
#                 except:
#                     value_types.add('text')

#         # Criteria for complex data:
#         # 1. Multiple basic types present
#         # 2. Special formats (currency, percentage) present
#         # 3. Multiple format variations within same type
#         # 4. Mixed numeric and text data
#         return (len(value_types) > 1 or 
#                 'currency' in value_types or 
#                 'percentage' in value_types or 
#                 format_variations > 2)

#     except Exception as e:
#         logger.warning(f"Error checking complex data for {column_name}: {str(e)}")
#         return False






def optimize_dataframe(df):
    # memory usage before optimization
    # culumn optimization
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        
        # if unique values are less than 50% of total values
        # convert to category
        if num_unique / len(df) < 0.5:
            df[col] = df[col].astype('category')
            
    # columns
    for col in df.select_dtypes(include=['int64']).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        # smallest possible integer type by range
        if col_min >= 0:
            if col_max < 255:
                df[col] = df[col].astype(np.uint8)
            elif col_max < 65535:
                df[col] = df[col].astype(np.uint16)
        else:
            if col_min > -128 and col_max < 127:
                df[col] = df[col].astype(np.int8)
            elif col_min > -32768 and col_max < 32767:
                df[col] = df[col].astype(np.int16)
                

    # float columns optimization
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
        
    return df

def generate_file_id():
    # generate a unique file ID
    return str(uuid.uuid4())

def save_dataframe(file_id, df):
    # dataframe to caches
    cache.set(file_id, pickle.dumps(df), timeout=3600)  # one hour cache expiry

def get_dataframe(file_id):
    # get dataframe from cache
    df_bytes = cache.get(file_id)
    if df_bytes is None:
        raise ValueError('File not found or expired')
    return pickle.loads(df_bytes)

def generate_preview_data(df):
    # generate preview data
    # df is the input DataFrame
    # return a list of dictionaries containing the first 5 rows of data
    preview_data = []
    for _, row in df.head(5).iterrows():
        row_dict = {}
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                row_dict[col] = "No data available"
            elif isinstance(value, (np.datetime64, pd.Timestamp)):
                row_dict[col] = value.strftime('%d/%m/%Y')  # australia date format
            elif isinstance(value, (np.floating, float)):
                if np.isnan(value):
                    row_dict[col] = "No data available"
                else:
                    row_dict[col] = f"{float(value):.2f}" if value % 1 != 0 else str(int(value))
            elif isinstance(value, bool):
                row_dict[col] = str(int(value))  # 'Yes'/'No'
            else:
                row_dict[col] = str(value).strip()
        preview_data.append(row_dict)
        # debug print
        # print("!!!!")
        # print(preview_data)
    return preview_data

def generate_sample_value(value):
    # generate a consistent sample value display format
    if pd.isna(value):
        return "No data available"
    elif isinstance(value, (np.datetime64, pd.Timestamp)):
        return value.strftime('%d/%m/%Y')
    elif isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return "No data available"
        return f"{float(value):.2f}" if value % 1 != 0 else str(int(value))
    elif isinstance(value, (np.integer, int)):
        return str(int(value))
    elif isinstance(value, bool):
        # 1 or 0 for boolean values
        return '1' if value else '0'
    else:
        str_value = str(value).strip()
        return str_value if str_value else "No data available"

def get_column_sample(df, column):
    # sample value for specific column
    # return a formatted sample value
    # column str is the column name
    # str: formatted sample value
    try:
        non_null_values = df[column].dropna()
        if not non_null_values.empty:
            return generate_sample_value(non_null_values.iloc[0])
        return "No data available"
    except Exception as e:
        print(f"Error processing sample for column {column}: {str(e)}")
        return "No data available"


def clean_special_values(df: pd.DataFrame) -> pd.DataFrame:
    # specific value and format cleaning
    # df: pd.DataFrame: input DataFrame
    # return: pd.DataFrame: cleaned DataFrame
    try:
        # backup the original data to avoid modification
        # create a copy of the DataFrame to avoid modifying the original data
        df = df.copy()
        
        # null values mapping
        null_values: Dict[str, Any] = {
            'NA': pd.NA,
            'N/A': pd.NA,
            'null': pd.NA,
            'NULL': pd.NA,
            'None': pd.NA,
            '-': pd.NA,
            'missing': pd.NA,
            'MISSING': pd.NA,
            'undefined': pd.NA,
            'UNDEFINED': pd.NA,
            'not available': pd.NA,
            'NOT AVAILABLE': pd.NA,
            'nan': pd.NA,
            'NaN': pd.NA,
            '': pd.NA,
        }
        
        # invalid numbers
        invalid_numbers: List[str] = ['#REF!', '#VALUE!', '#DIV/0!', '#NUM!']
        for num in invalid_numbers:
            null_values[num] = pd.NA
            
        # replace null values
        df = df.replace(null_values)
        
        # mixed date-time values
        def clean_datetime(value: Any) -> Any:
            # mxied date-time values
            if pd.isna(value):
                return value
                
            try:
                value = str(value).strip()
                
                # date-time combined patterns
                patterns = [
                    # "2024-03-12 14:30:00"
                    (r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})', r'\1'),
                    # "12/03/2024 2:30 PM"
                    (r'(\d{1,2}/\d{1,2}/\d{4})\s+\d{1,2}:\d{2}\s*(?:AM|PM)?', r'\1'),
                    # "20240312"
                    (r'(\d{4})(\d{2})(\d{2})', r'\2/\3/\1'),  # MM/DD/YYYY
                    # "Mar 12, 2024"
                    (r'([A-Za-z]{3})\s+(\d{1,2}),\s*(\d{4})', lambda m: f"{m.group(2)}/{months[m.group(1).lower()]}/{m.group(3)}"),
                    # "12-Mar-2024"
                    (r'(\d{1,2})-([A-Za-z]{3})-(\d{4})', lambda m: f"{m.group(1)}/{months[m.group(2).lower()]}/{m.group(3)}"),
                ]
                
                # month name mapping
                months = {
                    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
                    'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
                    'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
                }
                
                # apply patterns
                for pattern, replacement in patterns:
                    if re.match(pattern, value):
                        value = re.sub(pattern, replacement, value)
                        break
                
                # try to convert to date format
                try:
                    # (20240312)
                    if value.isdigit() and len(value) == 8:
                        value = f"{value[4:6]}/{value[6:]}/{value[:4]}"
                    return value
                except:
                    return value
                    
            except Exception as e:
                logger.warning(f"Error cleaning datetime value '{value}': {str(e)}")
                return value
        
        # date-time columns
        date_columns = [col for col in df.columns 
            if any(date_term in col.lower() 
                    for date_term in ['date', 'time', 'day', 'year', 'month'])]
        
        for col in date_columns:
            logger.info(f"Cleaning datetime values in column: {col}")
            df[col] = df[col].apply(clean_datetime)
        
        # special characters in numeric columns
        numeric_cleaner = lambda x: str(x).replace('$', '').replace(',', '') if pd.notna(x) else x
        
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            df[col] = df[col].apply(numeric_cleaner)
            
        # % percentage columns
        def clean_percentage(value: Any) -> Any:
            if pd.isna(value):
                return value
            try:
                value = str(value)
                if '%' in value:
                    return float(value.replace('%', '')) / 100
                return value
            except:
                return value
                
        percentage_columns = [col for col in df.columns 
            if any(term in col.lower() 
                    for term in ['percent', 'percentage', 'rate', '%'])]
        
        for col in percentage_columns:
            df[col] = df[col].apply(clean_percentage)
            
        logger.info("Successfully cleaned special values in DataFrame")
        return df
        
    except Exception as e:
        logger.error(f"Error in clean_special_values: {str(e)}")
        raise

# given convert function
# basic data type inference
# def infer_and_convert_data_types(df):
#     # buffed up data type inference function
#     # clean special values first
#     df = clean_special_values(df)
#     print(1)
#     for column in df.columns:
#         try:

#             df_converted = pd.to_numeric(df[column], errors='coerce')
#             if not df_converted.isna().all():
#                 df[column] = df_converted
#                 continue
#             print(2)

#             try:
#                 df[column] = pd.to_datetime(df[column], errors='coerce', format='mixed')
#                 continue
#             except (ValueError, TypeError):
#                 pass
#             print(3)

#             if len(df[column].unique()) / len(df[column]) < 0.5:
#                 df[column] = pd.Categorical(df[column])
#                 continue

#             non_null_values = df[column].dropna().tolist()
#             if is_complex_data(non_null_values, column):
#                 inferred_type, confidence = SpacyModelCache.analyze_complex_data(
#                     non_null_values, column
#                 )
#                 if confidence > 0.5:
#                     if inferred_type == 'datetime':
#                         df[column] = pd.to_datetime(df[column], errors='coerce')
#                     elif inferred_type == 'number':
#                         df[column] = pd.to_numeric(df[column], errors='coerce')
#                     elif inferred_type == 'category':
#                         df[column] = pd.Categorical(df[column])

#             print(4)
#         except Exception as e:
#             logger.error(f"Error processing column {column}: {str(e)}")
#             df[column] = df[column].astype(str)
#         print(5)
#         return df


def infer_and_convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """集中处理所有类型转换逻辑，并添加详细日志"""
    df = df.copy()

    for column in df.columns:
        try:
            non_null_values = df[column].dropna().tolist()
            
            # 检查是否为复杂数据
            if is_complex_data(non_null_values, column):
                logger.info(f"Analyzing complex column: {column}")
                # 获取类型推断结果
                inferred_type, confidence = SpacyModelCache.analyze_complex_data(
                    non_null_values, column
                )
                
                logger.info(f"Type inference result for {column}: {inferred_type} (confidence: {confidence:.2f})")
                
                if confidence > 0.5:
                    try:
                        # 记录转换前的类型
                        original_type = str(df[column].dtype)
                        logger.info(f"Attempting to convert {column} from {original_type} to {inferred_type}")
                        
                        # 执行转换
                        df = convert_column_type(df, column, inferred_type)
                        
                        # 记录转换后的类型
                        new_type = str(df[column].dtype)
                        logger.info(f"Successfully converted {column} to {new_type}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error converting {column} to {inferred_type}: {str(e)}")
            
            # 基础推断逻辑
            logger.info(f"Using basic inference for {column}")
            original_type = str(df[column].dtype)
            
            # 尝试数值转换
            df_converted = pd.to_numeric(df[column], errors='coerce')
            if not df_converted.isna().all():
                df[column] = df_converted
                logger.info(f"Converted {column} to numeric: {str(df[column].dtype)}")
                continue

            # 尝试日期转换
            try:
                df[column] = pd.to_datetime(df[column])
                logger.info(f"Converted {column} to datetime: {str(df[column].dtype)}")
                continue
            except (ValueError, TypeError):
                pass

            # 检查分类
            if len(df[column].unique()) / len(df[column]) < 0.5:
                df[column] = pd.Categorical(df[column])
                logger.info(f"Converted {column} to category: {str(df[column].dtype)}")
                continue

            # 默认文本
            df[column] = df[column].astype(str)
            logger.info(f"Converted {column} to string: {str(df[column].dtype)}")

        except Exception as e:
            logger.error(f"Error processing column {column}: {str(e)}")
            df[column] = df[column].astype(str)
            logger.info(f"Fallback: converted {column} to string after error")

    return df

def convert_column_type(df: pd.DataFrame, column: str, inferred_type: str) -> pd.DataFrame:
    """统一的类型转换函数，添加详细日志"""
    try:
        original_type = str(df[column].dtype)
        logger.info(f"Converting {column} from {original_type} to {inferred_type}")
        
        if inferred_type == 'datetime':
            df[column] = pd.to_datetime(df[column], errors='coerce')
            logger.info(f"Date conversion result for {column}: {str(df[column].dtype)}")
        elif inferred_type == 'boolean':
            df[column] = df[column].map({
                '1': True, '0': False,
                True: True, False: False,
                1: True, 0: False
            }).astype('boolean')
            logger.info(f"Boolean conversion result for {column}: {str(df[column].dtype)}")
        elif inferred_type == 'category':
            df[column] = pd.Categorical(df[column])
            logger.info(f"Category conversion result for {column}: {str(df[column].dtype)}")
        elif inferred_type == 'number':
            df[column] = pd.to_numeric(df[column], errors='coerce')
            logger.info(f"Numeric conversion result for {column}: {str(df[column].dtype)}")
            
        return df
    except Exception as e:
        logger.error(f"Error in type conversion for {column}: {str(e)}")
        return df

def update_column_type(df: pd.DataFrame, column: str, new_type: str) -> pd.DataFrame:
    """
    Safe column type update function
    """
    try:
        df = df.copy()
        
        if new_type == 'category':
            df[column] = pd.Categorical(df[column])
        elif new_type == 'number':
            df[column] = pd.to_numeric(df[column], errors='coerce')
        elif new_type == 'datetime':
            df[column] = pd.to_datetime(df[column], errors='coerce')
        elif new_type == 'boolean':
            df[column] = df[column].map({'1': True, '0': False, 
                                       'true': True, 'false': False,
                                       'yes': True, 'no': False}).astype('boolean')
        else:
            df[column] = df[column].astype(str)
            
        return df
        
    except Exception as e:
        logger.error(f"Error updating column {column} to type {new_type}: {str(e)}")
        raise



    #         # get non-null values
    #         non_null_values = df[column].dropna().tolist()
    #         if len(non_null_values) == 0:
    #             df[column] = df[column].astype(str)
    #             continue

    #         # check if it's complex data
    #         if is_complex_data(non_null_values, column):
    #             logger.info(f"Complex data detected in column {column}, using spaCy model")
    #             inferred_type, confidence = SpacyModelCache.analyze_complex_data(
    #                 non_null_values, column
    #             )
                
    #             if confidence > 0.5:  # only use model inference if confidence is high enough
    #                 logger.info(f"Using model inference for {column}: {inferred_type} (confidence: {confidence:.2f})")
    #                 try:
    #                     if inferred_type == 'datetime':
    #                         df[column] = pd.to_datetime(df[column], errors='coerce')
    #                     elif inferred_type == 'number':
    #                         df[column] = pd.to_numeric(df[column], errors='coerce')
    #                     elif inferred_type == 'category':
    #                         df[column] = pd.Categorical(df[column])
    #                     continue
    #                 except Exception as e:
    #                     logger.warning(f"Failed to convert {column} using model inference: {str(e)}")
    #             else:
    #                 logger.info(f"Low confidence ({confidence:.2f}) for {column}, falling back to rule-based inference")

    #         # original rule-based inference logic
    #         col_lower = column.lower()
            
    #         # name-related columns are usually strings
    #         if any(name in col_lower for name in ['name', 'title', 'label', 'id']):
    #             df[column] = df[column].astype(str)
    #             continue
                
    #         # date-related columns
    #         if any(date in col_lower for date in ['date', 'time', 'year', 'month', 'day']):
    #             try:
    #                 df[column] = pd.to_datetime(df[column], format='%d/%m/%Y', errors='coerce')
    #                 continue
    #             except:
    #                 pass

    #         # grade columns
    #         if 'grade' in col_lower:
    #             unique_values = pd.Series(non_null_values).unique()
    #             if len(unique_values) <= 5 and all(len(str(x).strip()) <= 2 for x in unique_values):
    #                 df[column] = pd.Categorical(df[column])
    #                 continue
    #             else:
    #                 df[column] = df[column].astype(str)
    #                 continue

    #         # boolean columns
    #         if ('is_' in col_lower or 
    #             all(str(x).lower() in ['true', 'false', '1', '0', 'yes', 'no', 'y', 'n'] 
    #                 for x in non_null_values)):
    #             try:
    #                 bool_map = {
    #                     'true': True, 'false': False,
    #                     '1': True, '0': False,
    #                     'yes': True, 'no': False,
    #                     1: True, 0: False
    #                 }
    #                 df[column] = df[column].map(bool_map)
    #                 continue
    #             except:
    #                 pass

    #         # numeric columns check
    #         try:
    #             if all(str(x).replace('.', '').isdigit() or str(x).lower() in ['nan', 'not available', 'n/a', ''] 
    #                    for x in non_null_values):
    #                 if any('.' in str(x) for x in non_null_values):
    #                     df[column] = pd.to_numeric(df[column], errors='coerce')
    #                 else:
    #                     df[column] = pd.to_numeric(df[column], errors='coerce', downcast='integer')
    #                 continue
    #         except:
    #             pass

    #         # categorical columns check
    #         unique_ratio = len(pd.Series(non_null_values).unique()) / len(non_null_values)
    #         if unique_ratio < 0.5 and len(pd.Series(non_null_values).unique()) <= 10:
    #             df[column] = pd.Categorical(df[column])
    #             continue

    #         # default to string type
    #         df[column] = df[column].astype(str)

    #     except Exception as e:
    #         logger.error(f"Error processing column {column}: {str(e)}")
    #         df[column] = df[column].astype(str)  # default to string type in case of error

    # return df

