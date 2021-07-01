# import libraries
import pandas as pd
import numpy as np

# import full dataset
clean_df = pd.read_excel('./files/clean_dataset.xlsx')

# import definitions dataset
definition_df = pd.read_excel('./sector_master_definition.xlsx')
definition_df = definition_df[['Sector', 'Subsector', 'Archetype', 'Value Chain']]

# clean dataset
definition_df.fillna(' ', inplace=True)

empty = clean_df[clean_df[['Sector', 'Subsector', 'Archetype', 'Valuechain', 'Company Profile Information']].isnull().all(1)]
clean_df = pd.concat([clean_df, empty, empty]).drop_duplicates(keep=False)

clean_df['Sector'] = [i.upper() for i in clean_df['Sector'].values.astype(str).tolist()]
clean_df['Subsector'] = [i.upper() for i in clean_df['Subsector'].values.astype(str).tolist()]
clean_df['Archetype'] = [i.upper() for i in clean_df['Archetype'].values.astype(str).tolist()]
clean_df['Valuechain'] = [i.upper() for i in clean_df['Valuechain'].values.astype(str).tolist()]


# assign definition columns to variables and extract unique values
sector = sorted(list(set([i.upper() for i in definition_df['Sector'].values.tolist()])))


# break down the clean dataset
print(f'label\t\t\t\t\tcount\tpercent')
for label in sector:
    subset = clean_df[clean_df['Sector'] == label]
    subsector = sorted(list(set([i.upper() for i in definition_df['Subsector'][definition_df['Sector'] == label].values.tolist()])))
    archetype = sorted(list(set([i.upper() for i in definition_df['Archetype'][definition_df['Sector'] == label].values.tolist()])))
    valuechain = sorted(list(set([i.upper() for i in definition_df['Value Chain'][definition_df['Sector'] == label].values.tolist()])))

    print(f'{label:<4.4}\t\t\t\t\t{subset.shape[0]}\t{subset.shape[0] / clean_df.shape[0]:>7.3%}')
    for sublabel in subsector:
        sub_subset = subset[subset['Subsector'] == sublabel]
        print(f'> {sublabel:<38.37}{sub_subset.shape[0]}\t{sub_subset.shape[0] / clean_df.shape[0]:>7.3%}')
    print('-'*55)
