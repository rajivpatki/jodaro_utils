import os.path
import logging
_logs_gsheets = logging.getLogger(name=__name__)
import gspread
try:
    import pandas as pd
except:
    pass

def gs_read(spreadsheet_id, worksheet_name=None, range_name=None, pandas_df = True):
    # Authenticate with Google Sheets API
    gc = gspread.oauth(
        credentials_filename='_tokens/google_sheets_api.json',
        authorized_user_filename='_tokens/gs_token.json'
    )
    # Open the specified spreadsheet
    sh = gc.open_by_key(spreadsheet_id)
    # Get a list of worksheet names in the spreadsheet
    list_of_sheets = [_.title for _ in sh.worksheets()]
    # If the specified worksheet exists, open it. Otherwise, use the first sheet in the spreadsheet and log a warning message.
    if worksheet_name is not None and worksheet_name in list_of_sheets:
        ws = sh.worksheet(worksheet_name)
    else:
        ws = sh.sheet1
        _logs_gsheets.warning(f'{worksheet_name} was not found. Using {ws.title} instead')
    # Get all the data from the worksheet
    data = ws.get_all_records()
    # If pandas_df is True, return the data as a pandas DataFrame. Otherwise, return the data as a list of dictionaries.
    if pandas_df:
        return pd.DataFrame(data)
    else:
        return data

def gs_write(spreadsheet_id: str, worksheet_name: str, df: pd.DataFrame, append = False):
    """
    Writes a pandas.DataFrame to the specified Google Sheet.

    Args:
        spreadsheet_id (str): Spreadsheet ID (can be derived from the URL)
        worksheet_name (str): Name of the worksheet in which to write. If a worksheet with the specified name does not exist, the function will create one with the specified name.
        df (pd.DataFrame): Input dataframe
        append (bool, optional): Set to True to append data to existing data after reshape. Set to false to overwrite **all** data in the sheet. Defaults to False.
    """
    # Authenticate with Google Sheets API
    gc = gspread.oauth(
        credentials_filename='_tokens/google_sheets_api.json',
        authorized_user_filename='_tokens/gs_token.json'
    )

    # Open the specified spreadsheet
    sh = gc.open_by_key(spreadsheet_id)
    # Get a list of worksheet names in the spreadsheet
    list_of_sheets = [_.title for _ in sh.worksheets()]
    # If the specified worksheet exists, open it. Otherwise, create a new worksheet with the specified name.
    if worksheet_name in list_of_sheets:
        ws = sh.worksheet(worksheet_name)
    else:
        ws = sh.add_worksheet(title=worksheet_name, rows=len(df), cols=len(df.columns))
    # If append is True, reshape the input dataframe to match the columns of the sheet's data, replace NULL values with empty strings, and append the data to the sheet.
    if append:
        existing_columns = ws.get('1:1')
        existing_columns = pd.DataFrame(existing_columns, columns=existing_columns[0])
        df = df.reindex(columns=existing_columns.columns)
        df.fillna('', inplace=True)
        towrite = df.values.tolist()
        ws.append_rows(towrite)
    # If append is False, replace NULL values with empty strings, insert the column names as the first row of the data, clear the sheet, and write the data to the sheet.
    else:
        df.fillna('', inplace=True)
        towrite = df.values.tolist()
        towrite.insert(0, df.columns.to_list())
        ws.clear()
        ws.update(towrite)