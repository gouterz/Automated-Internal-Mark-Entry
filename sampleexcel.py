'''
from openpyxl import Workbook

xfile = openpyxl.load_workbook('sample.xls')

sheet = xfile.get_sheet_by_name('CSE')
sheet['A3'] = regno
'''
import pandas as pd
def append_df_to_excel(df, excel_path):
    df_excel = pd.read_excel(excel_path)
    df_excel = df_excel[df_excel.filter(regex='^(?!Unnamed)').columns]
    result = pd.concat([df_excel, df], ignore_index=True,sort=False)
    result.to_excel(excel_path, index=False)
df = pd.DataFrame({"Q9":[25],"Q3":[30],"Register Number":[20],"A":[30]})
df = df[df.filter(regex='^(?!Unnamed)').columns]
append_df_to_excel(df, r"sample.xls")


'''
import openpyxl
from openpyxl import Workbook
rb = openpyxl.load_workbook("sample.xlsx")
row_no = 3

rb.write(row_no,3,hello)
'''
