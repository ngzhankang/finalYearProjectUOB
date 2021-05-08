# import libraries
import spacy
import pandas as pd
from spacy.language import Language
from spacy.tokens import Doc
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

# set up spacy
spacy.prefer_gpu()

# init nlp
nlp = spacy.load('en_core_web_lg')

# ----- passthrough ----- #
# keywords masterlist
# 

# declare custom properties
Doc.set_extension('processed', default=True, force=True)
Doc.set_extension('word_bag', default=True, force=True)

# Modify tokenizer infix patterns
infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
)

infix_re = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_re.finditer

# custom lemmatizer
@Language.component("custom_preprocess")
def custom_preprocess(doc):
    temp = []

    # filter through each token and add to preprocessed text if requirements #
    # met.                                                                   #
    for t in doc:
        if (not t.is_punct and not t.like_num and not t.is_stop and not t.is_digit and not (t.ent_type == 396 or t.ent_type == 397)):
            temp.append(t.lemma_.upper())

    doc._.processed = temp

    return doc

# add custom pipeline components to default pipeline
nlp.add_pipe('custom_preprocess', last=True)



# === MAIN PREPROCESS FUNCTION === #
def preprocess(df, keywords):
    # clean dataset
    df.dropna(axis=0, how='any', subset=['Company Profile Information'], inplace=True)
    df['Company Profile Information'] = df['Company Profile Information'].astype(str)

    # do tokenisation on df
    processed_doc = list(nlp.pipe(df['Company Profile Information']))
    df['processed'] = [doc._.processed for doc in processed_doc]

    # perform bag of words
    bow_vectors = []
    for index, row in df.iterrows():
        company = row['processed']

        dictionary = dict.fromkeys(keywords, 0)
        for word in company:
            if word in keywords:
                dictionary[word] += 1

        # append to dataframe
        bow_vectors.append(list(dictionary.values()))

        # print(f'{sum(dictionary.values()):>3}/{len(dictionary.values()):<3} |', dictionary.values())

    df['BoW_vectors'] = bow_vectors

    return df