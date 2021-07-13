import pandas as pd
from bert_predict import predict

df = pd.read_excel('./files/input.xlsx')

# do prediction
df = predict(df)

# write results to excel file
df.to_excel('./files/output.xlsx', index=False)