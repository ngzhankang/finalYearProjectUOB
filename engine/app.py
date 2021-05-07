import pandas as pd
from predict import predict

df_test = pd.read_excel('engine/files/input.xlsx')

predict(df_test)