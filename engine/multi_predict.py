# import necessary libraries
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from preprocess import preprocess

# load models
model = load_model('./weights/multilabel_model')

# import sector master definition file
df_keywords = pd.read_excel('./sector_master_definition.xlsx')

# preprocess dataset
df_keywords.drop(['Explanations', 'Notes'], axis=1, inplace=True)
df_keywords['Value Chain'].fillna(' ', inplace=True)
df_keywords['Sector Keywords'].fillna('[]', inplace=True)
df_keywords['Sector Keywords'] = df_keywords['Sector Keywords'].str.upper()
df_keywords.dropna(axis=0, how='any', inplace=True)

# read and split classification tags
sectors = sorted(list(df_keywords['Sector'].str.upper().unique()))
subsectors = sorted(list(df_keywords['Subsector'].unique()))
archetypes = sorted(list(df_keywords['Archetype'].unique()))
valuechains = sorted(list(df_keywords['Value Chain'].str.upper().unique()))

class_counts = [len(sectors), len(subsectors), len(archetypes), len(valuechains)]
classes = [sectors, subsectors, archetypes, valuechains]

# build keyword master list
keywords = []
for index, item in df_keywords['Sector Keywords'].iteritems():
    keywords += eval(item)

keywords = sorted(list(set(keywords)))

# function for processing prediction results
def __process_results(result):
    temp = []

    for r in result:
        temp.append((np.argmax(r), r[np.argmax(r)]))

    return temp



# === MAIN PREDICT FUNCTION === #
def predict(df):
    df = preprocess(df, keywords)

    X_pred = np.array(list(df['BoW_vectors']))

    # do prediction
    results = model.predict(X_pred)

    # process output into rows
    processed = []
    for result in results:
        processed.append(__process_results(result))

    processed = np.array(processed)

    # print results in human readable form
    print('Prediction' + ' '*31 + '| Confidence')
    print('-'*53 + '\n')

    for row in processed.swapaxes(0, 1):
        for i in range(len(row)):
            print(f'{classes[i][int(row[i][0])]: <40.40} | {row[i][1] * 100: >9.4}%')
        
        print('\n' + '-'*22 + '\n')

    # add results to df

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

    return df.drop(['BoW_vectors', 'processed'], axis=1)