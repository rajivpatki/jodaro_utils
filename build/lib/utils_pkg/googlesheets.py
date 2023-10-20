import os
import site
from dotenv import load_dotenv
load_dotenv("python.env")
load_dotenv('.env')
site.addsitedir(str(os.getenv("PYTHONPATH")))

import os.path
import gspread
import pandas as pd

import logging
_logs_gsheets = logging.getLogger(name=__name__)

def gs_read(spreadsheet_id, worksheet_name=None, range_name=None):
    gc = gspread.oauth(
        credentials_filename='_tokens/google_sheets_api.json',
        authorized_user_filename='_tokens/gs_token.json'
    )
    sh = gc.open_by_key(spreadsheet_id)
    list_of_sheets = [_.title for _ in sh.worksheets()]
    if worksheet_name is not None and worksheet_name in list_of_sheets:
        ws = sh.worksheet(worksheet_name)
    else:
        ws = sh.sheet1
        _logs_gsheets.warning(f'{worksheet_name} was not found. Using {ws.title} instead')

    return pd.DataFrame(ws.get_all_records())

def gs_write(spreadsheet_id: str, worksheet_name: str, df: pd.DataFrame, append = False):
    """Writes a pandas.DataFrame to the specified spreadsheet_id/worksheet_name. If `append` is set to `True`, the input dataframe is reshaped to match the columns of the sheet's data. NULL values are replaces with empty strings

    Args:
        spreadsheet_id (str): Spreadsheet ID (can be derived from the URL)
        worksheet_name (str): name of the worksheet in which to write. If a worksheet with the specified name does not exists, the function will create one with the specified name
        df (pd.DataFrame): input dataframe
        append (bool, optional): Set to True to append data to existing data after reshape. Set to false to overwrite **all** data in the sheet. Defaults to False.
    """
    gc = gspread.oauth(
        credentials_filename='_tokens/google_sheets_api.json',
        authorized_user_filename='_tokens/gs_token.json'
    )
    sh = gc.open_by_key(spreadsheet_id)
    list_of_sheets = [_.title for _ in sh.worksheets()]
    if worksheet_name in list_of_sheets:
        ws = sh.worksheet(worksheet_name)
    else:
        ws = sh.add_worksheet(title=worksheet_name, rows=len(df), cols=len(df.columns))

    if append:
        existing_columns = ws.get('1:1')
        existing_columns = pd.DataFrame(existing_columns, columns=existing_columns[0])
        df = df.reindex(columns=existing_columns.columns)
        df.fillna('', inplace=True)
        towrite = df.values.tolist()
        ws.append_rows(towrite)
    else:
        df.fillna('', inplace=True)
        towrite = df.values.tolist()
        towrite.insert(0, df.columns.to_list())
        ws.clear()
        ws.update(towrite)