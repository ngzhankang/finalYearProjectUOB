# import necessary libraries
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from preprocess import preprocess

# load models
sector_model = load_model('engine/weights/model_1')
subsector_model = load_model('engine/weights/model_2')
archetype_model = load_model('engine/weights/model_3')
valuechain_model = load_model('engine/weights/model_4')

models = [sector_model, subsector_model, archetype_model, valuechain_model]

# import sector master definition file
df_keywords = pd.read_excel('engine/sector_master_definition.xlsx')

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

# build keyword master list
keywords = []
for index, item in df_keywords['Sector Keywords'].iteritems():
    keywords += eval(item)

keywords = sorted(list(set(keywords)))



# === MAIN PREDICT FUNCTION === #
def predict(df):
    df = preprocess(df, keywords)

    X_pred = np.array(list(df['BoW_vectors']))

    # do prediction
    results = []
    for model in models:
        results.append(model.predict(X_pred))

    print(results)

df_test = pd.read_excel('engine/files/input.xlsx')