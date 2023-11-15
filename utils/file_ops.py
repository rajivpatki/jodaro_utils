from functools import lru_cache
import os
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor
from utils.common_functions import list_distribute_into_blocks
import json
import logging
_logs_file_ops = logging.getLogger(__name__)

try:
    import boto3
    from botocore.errorfactory import ClientError
    from dotenv import load_dotenv
    load_dotenv("python.env")
    load_dotenv(".env")
    s3 = boto3.client('s3')
    s3_resource = boto3.resource('s3')
except Exception as e:
    _logs_file_ops.warning(f's3 fileops may not work: {e}')

try:
    import pandas as pd
    import numpy as np
    import pyarrow.parquet as pq
    from pyarrow.lib import ArrowInvalid
    from tqdm import tqdm
except Exception as e:
    _logs_file_ops.warning(f'Failed imports: {e}')

@lru_cache(maxsize=None)
def _bucket_prefix(path):
    """Splits the path to an s3 URI into its bucket name and prefix
    """
    bucket_name = re.search(r's3://([^/]+)/', path).group(1)
    prefix = re.search(r's3://[^/]+/(.+?)/?$', path).group(1)
    return bucket_name, prefix

def list_s3(folder_path):
    """Returns the full path to the contents of a specified s3 location
    """
    bucket, prefix = _bucket_prefix(folder_path)
    res = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    res = res.get('Contents', None)
    if res:
        return [f"s3://{bucket}/{_['Key']}" for _ in res]
    else:
        return []

def filter_paths(paths, filters=None):
    if filters is None:
        return paths

    import re
    from datetime import datetime
    def validate_operator(op):
        # Regex to match only valid comparison operators
        if not re.match(r'^[\<\>\=\!]+$', op):
            raise ValueError(f"Invalid operator: {op}")

    def apply_filters(path, filters):
        for key, op, value in filters:
            validate_operator(op)
            # Extract the value from the path using regular expression
            match = re.search(fr'{key}=([^/]+)', path)
            if not match:
                return False
            extracted_value = match.group(1)

            # Convert to date if value is in date format
            if re.match(r'\d{4}-\d{2}-\d{2}', value):
                extracted_value = datetime.strptime(extracted_value, '%Y-%m-%d')
                value = datetime.strptime(value, '%Y-%m-%d')
                comparison_string = f"extracted_value {op} value"
            else:
                comparison_string = f"'{extracted_value}' {op} '{value}'"

            # Evaluate the comparison using eval
            if not eval(comparison_string):
                return False
        return True

    # Apply filters to paths
    return [path for path in paths if apply_filters(path, filters)]

def listdir(folder_path: str, recurse=False, filters=None) -> list[str]:
    """
    Returns a list of full paths to the contents of a specified location, accepts s3

    Args:
    - folder_path (str): path to the folder to list contents of

    Returns:
    - list of str: list of full paths to the contents of the specified location
    """
    if 's3://' in folder_path:
        folder_path = folder_path.replace('\\', '/')
        if folder_path[-1] != '/':
            folder_path += '/'
        # create S3 client
        s3 = boto3.client('s3')
        # specify bucket and prefix (folder path)
        bucket_name, prefix = _bucket_prefix(folder_path)
        if prefix[-1] != '/':
            prefix += '/'
        def get_keys(bucket_name, prefix, recurse=False):
            keys = []
            kwargs = dict(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
            while True:
                resp = s3.list_objects_v2(**kwargs)
                files, folders = resp.get('Contents', []), resp.get('CommonPrefixes', [])
                for obj in files:
                    keys.append(obj['Key'])
                for obj in folders:
                    if recurse:
                        keys += get_keys(bucket_name, obj['Prefix'], recurse)
                    else:
                        keys.append(obj['Prefix'])
                try:
                    kwargs['ContinuationToken'] = resp['NextContinuationToken']
                except KeyError:
                    break
            # _logs_file_ops.info(f'{len(files)} files + {len(folders)} folders in {bucket_name}/{prefix}')
            return keys

        keys = get_keys(bucket_name=bucket_name, prefix=prefix, recurse=recurse)
        paths = [f's3://{bucket_name}/{_}' for _ in keys]
        if folder_path in paths:
            paths.remove(folder_path)
    else:
        paths = [os.path.join(folder_path, _) for _ in os.listdir(folder_path)]
    return filter_paths(paths=paths, filters=filters)

def touch_file(file_path: str) -> None:
    """Creates an empty file if it does not exist
    """
    if 's3://' in file_path:
        file_path = file_path.replace('\\', '/')
        bucket_name, prefix = _bucket_prefix(file_path)

        # Extract key (object key)
        key = re.search(r's3://[^/]+/(.+)', file_path).group(1)
        s3.put_object(
            Bucket=bucket_name,
            Key=key
        )
    else:
        with open(file_path, 'w') as document:
            pass

def touch_folder(folder_path:str) -> None:
    """Creates an empty folder if a folder of the path does not exists
    """
    if 's3://' in folder_path:
        return None
    elif not os.path.exists(folder_path):
        os.makedirs(folder_path)

def path_join(*args):
    """Joins paths such as folders and file names into a valid path string
    """
    joined = os.path.join(*args)
    if 's3://' in joined:
        return joined.replace('\\', '/')
    else:
        return joined

def read_from_json(path):
    """Reads to a dict a json file if it exists, else will return None
    """
    if 's3:' in path:
        bucket, prefix = _bucket_prefix(path)
        content_object = s3_resource.Object(bucket, prefix)
        try:
            file_content = content_object.get()['Body'].read().decode('utf-8')
            json_content = json.loads(file_content)
        except ClientError as e:
            if e.response['Error']['Code'] == "NoSuchKey":
                _logs_file_ops.info(f"Key does not exist!")
            else:
                _logs_file_ops.error(f"Error accessing Key: {e.response['Error']}")
            json_content = None

    elif os.path.exists(path):
        with open(path) as f_in:
            json_content = json.load(f_in)

    else:
        json_content = None
    return json_content

def write_to_json(data_dict, path):
    """Writes a dict or list[dict] to the specified file as a valid JSON
    """
    if 's3://' in path:
        bucket, prefix = _bucket_prefix(path)
        s3object = s3_resource.Object(bucket, prefix)

        s3object.put(
            Body=(bytes(json.dumps(data_dict).encode('UTF-8')))
        )
    else:
        with open(path, 'w') as f:
            json.dump(data_dict, f, ensure_ascii=False)

def write_file(data, path: str):
    """Writes bytes data to the path (key) provided for s3 or local storage based on the key.
    Do NOT use this to write dicts to JSON. Use `write_to_json` instead

    Args:
        data (bytes): Bytes object
        path (str): Storage file path
    """
    if 's3://' in path:
        bucket, prefix = _bucket_prefix(path)
        s3object = s3_resource.Object(bucket, prefix)
        s3object.put(Body=data)
        # s3.put_object(Body=data, Bucket=bucket, Key=prefix)
    else:
        with open(path, 'wb') as handle:
            handle.write(data)

def read_file(path: str):
    """Reads bytes data to the path (key) provided for s3 or local storage based on the key

    Args:
        path (str): Storage file path
    """
    if not path:
        _logs_file_ops.warning('Path was None')
        return None
    if 's3://' in path:
        bucket, prefix = _bucket_prefix(path)
        s3object = s3_resource.Object(bucket, prefix)
        data = s3object.get()['Body'].read()
        # s3.put_object(Body=data, Bucket=bucket, Key=prefix)
    else:
        with open(path, 'rb') as handle:
            data = handle.read()
    return data

def read_parquets(
        path = None, columns = None, threads:int=10, max_retries:int = 5, filters = None
    ):
    """Reads the specified path (s3 or local) as a pandas dataframe

    Args:
        path (str | list[str]): A folder or a list of parquet files
        - If path the function will look for .parquet files and concat them
        - If a list of paths, the function will filter .parquet files and concat them

        columns (list[str] | None, optional): Specify columns to be read from the parquet file
        - Specify columns to reduce the size of data being transfered and stored in memory
        - If `None`, all columns will be read
        
        threads (int, optional): Number of parallel threads reading
        - Do NOT exceed 10 for s3, else you may hit the rate limit
        
        max_retries (int, optional): Max times to retry
        replace_npnan (bool, !DEPRECATED): Replace nan values?

    Returns:
        pandas.DataFrame: A pandas dataframe with all the parquet files concatenated
    """

    # If the path is not an s3 path, set the number of threads to 500
    if not any(['s3:' in path, any(['s3:' in _ for _ in path])]):
        threads=500

    def _read_parquet_file(file_path, columns=None, retry_number = 0):
        """Read a single parquet file
        """
        # If the number of retries exceeds the max_retries, log an error and return None
        if retry_number >= max_retries:
            _logs_file_ops.error(f'Failed for {file_path} after {retry_number} retries')
            return None
        try:
            # Read the parquet file
            parquet_file  = pq.ParquetFile(file_path)
            # If the number of rows in the file is 0, return None
            if parquet_file.metadata.num_rows == 0:
                return None
            # If columns are specified, only read those columns
            if columns is not None:
                columns = [_ for _ in columns if _ in parquet_file.schema_arrow.names]
                if len(columns)==0: return None
            # Return the pandas dataframe
            return (parquet_file.read(columns=columns).to_pandas())
        except Exception as e:
            # If the exception is an ArrowInvalid, log a warning and return None
            if isinstance(e, ArrowInvalid):
                _logs_file_ops.warning(f'Skipping invalid parquet file: {file_path}')
                return None
            # If the exception is not an ArrowInvalid, log a warning and retry
            _logs_file_ops.warning(f'Retrying: {file_path}: {e}')
            # Full jitter wait before retry
            time.sleep(min(10, 2**retry_number)/2 + random.uniform(0, min(10, 2**retry_number)))
            return _read_parquet_file(file_path, columns, retry_number+1)

    def _read_list_of_parquet_files(file_list: list[str], columns=None):
        """Read a list of parquet files serially using `_read_parquet_file`
        """
        # Concatenate all the parquet files in the list into a single pandas dataframe
        return pd.concat([
                _read_parquet_file(file_path=f, columns=columns)
                for f in tqdm(file_list, desc = 'Reading...', leave = False)
            ]).reset_index(drop=True)

    # If the path is a string, check if it's a parquet file or a folder
    if isinstance(path, str):
        if '.parquet' in path:
            # If it's a parquet file, read it and return the pandas dataframe
            list_of_files = [path]
        else:
            # If it's a folder, get a list of all the parquet files in the folder
            list_of_files = [
                _ for _ in listdir(folder_path=path, recurse=True, filters=filters)
                if '.parquet' in _
            ]
    # If the path is a list, use the list as the list of parquet files
    elif isinstance(path, list):
        list_of_files = [_ for _ in filter_paths(paths=path, filters=filters) if '.parquet' in _]

    # If there are no parquet files in the list, log a warning and return None
    if len(list_of_files) == 0:
        _logs_file_ops.warning(f'No .parquet files found in {path}')
        return None
    # If threads is specified, distribute the list of parquet files into blocks and read them in parallel
    if threads:
        file_list_of_lists = list_distribute_into_blocks(list_of_files, threads)
    # If threads is not specified, read the list of parquet files serially
    else:
        file_list_of_lists = [[_] for _ in list_of_files]
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(_read_list_of_parquet_files, file_list, columns)
            for file_list in file_list_of_lists
        ]
        sublists = [future.result() for future in futures]

    # Concatenate all the sublists into a single pandas dataframe and return it
    return pd.concat(sublists).reset_index(drop=True)