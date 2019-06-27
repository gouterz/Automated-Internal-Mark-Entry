# importing xlwt module 
import xlwt 

workbook = xlwt.Workbook() 

sheet = workbook.add_sheet("CSE") 

# Specifying style 
style = xlwt.easyxf('font: bold 1') 

# Specifying column 
sheet.write(0, 0, 'Register Number', style)
sheet.write(0, 1, 'Q1', style)
sheet.write(0, 2, 'Q2', style)
sheet.write(0, 3, 'Q3', style)
sheet.write(0, 4, 'Q4', style)
sheet.write(0, 5, 'Q5', style)
sheet.write(0, 6, 'Q6', style)
sheet.write(0, 7, 'Q7', style)
sheet.write(0, 8, 'Q8', style)
sheet.write(0, 9, 'Q9', style)
sheet.write(0, 10, 'Q10', style)
sheet.write(0, 11, 'Q11a', style)
sheet.write(0, 12, 'Q11b', style)
sheet.write(0, 13, 'Q12a', style)
sheet.write(0, 14, 'Q12b', style)
sheet.write(0, 15, 'Q13a', style)
sheet.write(0, 16, 'Q13b', style)
sheet.write(0, 17, 'Q14a', style)
sheet.write(0, 18, 'Q14b', style)
sheet.write(0, 19, 'Q15a', style)
sheet.write(0, 20, 'Q15b', style)
workbook.save("sample.xls") 
