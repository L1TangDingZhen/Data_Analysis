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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpacyModelCache:
    _instance = None
    _model = None

    @classmethod
    def get_model(cls) -> spacy.language.Language:
        # load spacy model
        if cls._model is None:
            try:
                cls._model = spacy.load('en_core_web_md')
                logger.info("Successfully loaded spaCy model")
            except Exception as e:
                logger.error(f"Error loading spaCy model: {str(e)}")
                raise
        return cls._model

    @classmethod
    def analyze_complex_data(cls, 
            values: List[Any], 
            column_name: str) -> Tuple[str, float]:
        
        # complex data analysis
        # values is a list of values in the column
        # column_name is the name of the column
        # return the inferred type and confidence (type and confidence)
        try:
            model = cls.get_model()
            predictions = []
            confidence_scores = []
            
            # 10 non-null values for analysis
            sample_values = [str(v) for v in values[:10] if pd.notna(v)]
            
            for value in sample_values:
                doc = model(value)
                
                # get entity type
                if doc.ents:
                    ent = doc.ents[0]
                    predictions.append(cls._map_entity_to_type(ent.label_))
                    confidence_scores.append(ent._.trf_score if hasattr(ent._, 'trf_score') else 0.5)
                else:
                    # get type from POS
                    pos_type = cls._get_type_from_pos(doc)
                    if pos_type:
                        predictions.append(pos_type)
                        confidence_scores.append(0.3)  # low confidence
            
            if predictions:
                # most common prediction
                most_common = max(set(predictions), key=predictions.count)
                confidence = sum(c for p, c in zip(predictions, confidence_scores) 
                    if p == most_common) / len(predictions)
                return most_common, confidence
            
            return 'text', 0.0  # return text type if no entities found
            
        except Exception as e:
            logger.warning(f"Error in complex data analysis: {str(e)}")
            return 'text', 0.0
            
    @staticmethod
    def _map_entity_to_type(ent_label: str) -> str:
        # entity label to data type mapping
        type_mapping = {
            'DATE': 'datetime',
            'TIME': 'datetime',
            'CARDINAL': 'number',
            'MONEY': 'number',
            'PERCENT': 'number',
            'QUANTITY': 'number',
            'ORDINAL': 'number'
        }
        return type_mapping.get(ent_label, 'text')
        
    @staticmethod
    def _get_type_from_pos(doc: spacy.tokens.Doc) -> Optional[str]:
        # predict type based on POS
        # main POS tag
        main_pos = doc[0].pos_
        
        if main_pos in ['NUM']:
            return 'number'
        elif main_pos in ['PROPN', 'NOUN']:
            return 'category'
            
        return None


# complex data detection
def is_complex_data(values: List[Any], column_name: str) -> bool:
    # check if the column contains complex data
    # return True if complex data is detected, False otherwise
    try:
        # non null values
        clean_values = [v for v in values if pd.notna(v)]
        if not clean_values:
            return False
            
        # first 10 values
        value_types = set()
        for value in clean_values[:10]:  # check first 10 values
            # if it's a pandas Timestamp
            if isinstance(value, pd.Timestamp):
                value_types.add('date')
            # if it's a number
            elif isinstance(value, (int, float)):
                value_types.add('number')
            elif isinstance(value, str):
                # if it's a string, check if it can be converted to other types
                try:
                    float(value)
                    value_types.add('number')
                    continue
                except ValueError:
                    pass
                
                # convert to datetime
                try:
                    pd.to_datetime(value)
                    value_types.add('date')
                    continue
                except:
                    pass
                
                # check if it's a currency format (e.g., $1,234.56)
                if re.match(r'^\$?\d{1,3}(,\d{3})*(\.\d+)?$', value):
                    value_types.add('currency')
                    continue
                    
                # e.g., 10%, 10.5%, 0.5%
                if re.match(r'^\d+(\.\d+)?%$', value):
                    value_types.add('percentage')
                    continue
                
                # other cases are considered as text
                value_types.add('text')
        
        # multiple types or special types (currency, percentage) are considered complex
        special_types = {'currency', 'percentage'}
        return len(value_types) > 1 or bool(value_types.intersection(special_types))
            
    except Exception as e:
        logger.warning(f"Error checking complex data: {str(e)}")
        return False



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
def infer_and_convert_data_types(df):
    # buffed up data type inference function
    # clean special values first
    df = clean_special_values(df)
    
    for column in df.columns:
        try:
            # get non-null values
            non_null_values = df[column].dropna().tolist()
            if len(non_null_values) == 0:
                df[column] = df[column].astype(str)
                continue

            # check if it's complex data
            if is_complex_data(non_null_values, column):
                logger.info(f"Complex data detected in column {column}, using spaCy model")
                inferred_type, confidence = SpacyModelCache.analyze_complex_data(
                    non_null_values, column
                )
                
                if confidence > 0.5:  # only use model inference if confidence is high enough
                    logger.info(f"Using model inference for {column}: {inferred_type} (confidence: {confidence:.2f})")
                    try:
                        if inferred_type == 'datetime':
                            df[column] = pd.to_datetime(df[column], errors='coerce')
                        elif inferred_type == 'number':
                            df[column] = pd.to_numeric(df[column], errors='coerce')
                        elif inferred_type == 'category':
                            df[column] = pd.Categorical(df[column])
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to convert {column} using model inference: {str(e)}")
                else:
                    logger.info(f"Low confidence ({confidence:.2f}) for {column}, falling back to rule-based inference")

            # original rule-based inference logic
            col_lower = column.lower()
            
            # name-related columns are usually strings
            if any(name in col_lower for name in ['name', 'title', 'label', 'id']):
                df[column] = df[column].astype(str)
                continue
                
            # date-related columns
            if any(date in col_lower for date in ['date', 'time', 'year', 'month', 'day']):
                try:
                    df[column] = pd.to_datetime(df[column], format='%d/%m/%Y', errors='coerce')
                    continue
                except:
                    pass

            # grade columns
            if 'grade' in col_lower:
                unique_values = pd.Series(non_null_values).unique()
                if len(unique_values) <= 5 and all(len(str(x).strip()) <= 2 for x in unique_values):
                    df[column] = pd.Categorical(df[column])
                    continue
                else:
                    df[column] = df[column].astype(str)
                    continue

            # boolean columns
            if ('is_' in col_lower or 
                all(str(x).lower() in ['true', 'false', '1', '0', 'yes', 'no'] 
                    for x in non_null_values)):
                try:
                    bool_map = {
                        'true': True, 'false': False,
                        '1': True, '0': False,
                        'yes': True, 'no': False,
                        1: True, 0: False
                    }
                    df[column] = df[column].map(bool_map)
                    continue
                except:
                    pass

            # numeric columns check
            try:
                if all(str(x).replace('.', '').isdigit() or str(x).lower() in ['nan', 'not available', 'n/a', ''] 
                       for x in non_null_values):
                    if any('.' in str(x) for x in non_null_values):
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                    else:
                        df[column] = pd.to_numeric(df[column], errors='coerce', downcast='integer')
                    continue
            except:
                pass

            # categorical columns check
            unique_ratio = len(pd.Series(non_null_values).unique()) / len(non_null_values)
            if unique_ratio < 0.5 and len(pd.Series(non_null_values).unique()) <= 10:
                df[column] = pd.Categorical(df[column])
                continue

            # default to string type
            df[column] = df[column].astype(str)

        except Exception as e:
            logger.error(f"Error processing column {column}: {str(e)}")
            df[column] = df[column].astype(str)  # default to string type in case of error

    return df


