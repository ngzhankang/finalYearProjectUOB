import pandas as pd
from multi_predict import predict

df_test = pd.read_excel('./files/input.xlsx')

# do prediction
df_test = predict(df_test)

# write results to excel file
df_test.to_excel('./files/output.xlsx', index=False)