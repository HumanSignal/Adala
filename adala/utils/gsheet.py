import pandas as pd
import gspread
from google.auth import default


def get_client():
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    credentials, project_id = default(scopes)
    return gspread.authorize(credentials)


def read_gsheet(url, sheet_name):
    client = get_client()

    workbook = client.open_by_url(url)
    sheet = workbook.worksheet(sheet_name)

    data = sheet.get_all_records()
    return pd.DataFrame(data)


def write_gsheet(df, url, sheet_name):
    client = get_client()

    spreadsheet = client.open_by_url(url)
    try:
        worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=str(df.shape[0]), cols=str(df.shape[1]))
    except gspread.exceptions.APIError:
        worksheet = spreadsheet.worksheet(sheet_name)
        spreadsheet.del_worksheet(worksheet)
        worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=str(df.shape[0]), cols=str(df.shape[1]))

    worksheet.update([df.columns.values.tolist()] + df.values.tolist())


def update_gsheet(df, url, sheet_name):
    client = get_client()
    spreadsheet = client.open_by_url(url)
    worksheet = spreadsheet.worksheet(sheet_name)
    existing_columns = worksheet.row_values(1)
    df_columns = df.columns.tolist()
    cell_updates = []

    for col in df_columns:
        if col in existing_columns:
            col_index = gspread.utils.a1_to_rowcol(f'{col}1')[1]
        else:
            col_index = len(existing_columns) + 1
            existing_columns.append(col)
            cell_updates.append(gspread.Cell(1, col_index, col))

        for i, value in enumerate(df[col], start=2):
            cell_updates.append(gspread.Cell(i, col_index, value))

    print('Updating cells...')
    worksheet.update_cells(cell_updates)