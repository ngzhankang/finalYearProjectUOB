import pandas as pd
from predict import predict

df_test = pd.read_excel('engine/files/input.xlsx')

# do prediction
df_test = predict(df_test)

# write results to excel file
df_test.to_excel('engine/files/output.xlsx', index=False)