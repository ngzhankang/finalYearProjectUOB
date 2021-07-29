from os import sep
import pandas as pd
from bert_predict import predict

print('Loading input...', end='    ')
try:
    df = pd.read_excel('./files/input.xlsx')
    print('Done!')

except FileNotFoundError:
    print('No input file found!')
    quit()

# do prediction
df, dropped_rows = predict(df)

# write results to excel file
df.to_excel('./files/output.xlsx', index=False)
print('Done!')

# print final message
record_count = df.shape[0]
print(f'\n{record_count - len(dropped_rows)}/{record_count} rows predicted. Row(s)', end=' ')
print(*dropped_rows, sep=', ', end=' ')
print('dropped as they were invalid.')