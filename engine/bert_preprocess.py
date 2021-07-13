# import libraries
from transformers import BertConfig, BertTokenizerFast

# set up BERT tokeniser
config = BertConfig.from_pretrained('bert-base-uncased')
config.output_hidden_states = False

tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path='bert-base-uncased', config=config)

# === MAIN PREPROCESS FUNCTION === #
def preprocess(df):
    # clean dataset
    df.dropna(axis=0, how='any', subset=['Company Profile Information'], inplace=True)
    df['Company Profile Information'] = df['Company Profile Information'].astype(str)

    # do BERT tokenisation on dataset
    bert_tokens = tokenizer(text=df['Company Profile Information'].to_list(),
                    add_special_tokens=True, # add special tokens like [SEP] and others
                    max_length=100,  # this is the max length of the sentence-to-be-token
                    truncation=True,
                    padding=True,
                    return_tensors='tf', # to return it as tf tensors to feed into keras API
                    return_token_type_ids=False,
                    return_attention_mask=True, # generate attention mask
                    verbose=True)

    tokens_labels = bert_tokens.copy()

    # return BERT tokens
    return dict(tokens_labels)