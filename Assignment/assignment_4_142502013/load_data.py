import pandas as pd


excel_file = "Online Retail.xlsx"
df = pd.read_excel(excel_file)
df_1000 = df.head(1000)


output_file = "online_retail_1000.csv"
df_1000.to_csv(output_file, index=False)

print(f"saved first 1000 records from {excel_file} into {output_file}")
