# import necessary libraries
import pandas as pd
import numpy as np
from bert_preprocess import preprocess

# do not print warnings
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model

# load models
model = load_model('./weights/bert')

# import sector master definition file
print('\n\nLoading Labels...', end='    ')
df_keywords = pd.read_excel('./sector_master_definition.xlsx')
print('Done!')

# preprocess definitions
print('Processing Labels...', end='    ')
df_keywords.drop(['Explanations', 'Notes'], axis=1, inplace=True)
df_keywords['Value Chain'].fillna(' ', inplace=True)
df_keywords.dropna(axis=0, how='any', inplace=True)
print('Done!')

# read and split classification tags
print('Building Labels...', end='    ')
sectors = sorted(list(df_keywords['Sector'].str.upper().unique()))
subsectors = sorted(list(df_keywords['Subsector'].unique()))
archetypes = sorted(list(df_keywords['Archetype'].unique()))
valuechains = sorted(list(df_keywords['Value Chain'].str.upper().unique()))
print('Done!')

class_counts = [len(sectors), len(subsectors), len(archetypes), len(valuechains)]
classes = [sectors, subsectors, archetypes, valuechains]

# function for processing prediction results
def __process_results(result):
    temp = []

    for r in result:
        temp.append((np.argmax(r), r[np.argmax(r)]))

    return temp



# === MAIN PREDICT FUNCTION === #
def predict(df):
    print('Preprocessing input...', end='    ')
    X_pred, dropped_rows = preprocess(df)
    print('Done!')

    # do prediction
    print('Performing prediction...', end='    ')
    results = model.predict(X_pred)
    print('Done!')

    # process output into rows
    print('Processing predictions...', end='    ')
    processed = []
    for result in results:
        processed.append(__process_results(result))

    processed = np.array(processed)
    print('Done!')

    # # print results in human readable form
    # print('Prediction' + ' '*31 + '| Confidence')
    # print('-'*53 + '\n')

    # for index, row in enumerate(processed.swapaxes(0, 1)):
    #     print('Company:', df['Company'].iloc[index], '\n')

    #     for i in range(len(row)):
    #         print(f'{classes[i][int(row[i][0])]: <40.40} | {row[i][1] * 100: >9.4}%')
        
    #     print('\n' + '-'*22 + '\n')

    # add results to df
    print('Writing predictions...', end='    ')
    processed_tags = []
    for i, result in enumerate(processed):
        temp = []
        for j, _ in result:
            temp.append(classes[i][int(j)])

        processed_tags.append(temp)

    df['Sector'] = processed_tags[0]
    df['Subsector'] = processed_tags[1]
    df['Archetype'] = processed_tags[2]
    df['Valuechain'] = processed_tags[3]

    return df, dropped_rows