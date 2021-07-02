# import libraries
import pandas as pd
import numpy as np

# import full dataset
data_df = pd.read_excel('./files/clean_dataset.xlsx')

# import definitions dataset
definition_df = pd.read_excel('./sector_master_definition.xlsx')
definition_df = definition_df[['Sector', 'Subsector', 'Archetype', 'Value Chain']]

# clean dataset
definition_df.fillna(' ', inplace=True)

empty = data_df[data_df[['Sector', 'Subsector', 'Archetype', 'Valuechain', 'Company Profile Information']].isnull().all(1)]
clean_df = pd.concat([data_df, empty, empty]).drop_duplicates(keep=False)

clean_df['Sector'] = [i.upper() for i in clean_df['Sector'].values.astype(str).tolist()]
clean_df['Valuechain'] = [i.upper() for i in clean_df['Valuechain'].values.astype(str).tolist()]


# assign definition columns to variables and extract unique values
sector = sorted(list(set([i.upper() for i in definition_df['Sector'].values.tolist()])))


# declare some global vars
split_df = []
clean_stats = ''
sample_stats = ''

print(f'label\t\t\t\t\tcount\tpercent')
for label in sector:
    subset = clean_df[clean_df['Sector'] == label]
    subsector = sorted(list(set(definition_df['Subsector'][definition_df['Sector'] == label].values.tolist())))

    # split off 200 records from each sector
    split_df_sample = subset.sample(200)
    split_df.append(split_df_sample)

    # make headers for clean and sampled dataset statistics
    clean_stats += f'{label:<4.4}\t\t\t\t\t{subset.shape[0]}\t{subset.shape[0] / clean_df.shape[0]:>7.3%}\n'
    sample_stats += f'{label:<4.4}\t\t\t\t\t{split_df_sample.shape[0]}\t{split_df_sample.shape[0] / clean_df.shape[0]:>7.3%}\n'

    # generate dataset statistics
    for sublabel in subsector:
        sub_subset = subset[subset['Subsector'] == sublabel]
        sample_sub_subset = split_df_sample[split_df_sample['Subsector'] == sublabel]

        # append statistics to appropriate vars
        clean_stats += f'> {sublabel:<38.37}{sub_subset.shape[0]}\t{sub_subset.shape[0] / clean_df.shape[0]:>7.3%}\n'
        sample_stats += f'> {sublabel:<38.37}{sample_sub_subset.shape[0]}\t{sample_sub_subset.shape[0] / clean_df.shape[0]:>7.3%}\n'

    clean_stats += '-'*55 + '\n'
    sample_stats += '-'*55 + '\n'
    
# print stats
print(clean_stats)
print(f'{"Sampled Subset":^55.55}\n{"-"*55}')
print(sample_stats)

# combine and shuffle sampled data into dataframe
split_df = pd.concat(split_df).sample(frac=1)

# remove sampled data from clean dataset
data_df.drop(split_df.index, inplace=True)

# write split to excel files
split_df.to_excel('./files/val_dataset.xlsx', index=False)
data_df.to_excel('./files/split_dataset.xlsx', index=False)