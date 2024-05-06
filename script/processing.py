import os
import re
import string
from langid.langid import LanguageIdentifier, model
import pandas as pd

def read_data_from_dir(data_dir):
    data = []
    for folder in os.listdir(data_dir):
        for file in os.listdir(data_dir + '/' + folder):
            with open(data_dir + '/' + folder + '/' + file, 'r', encoding="utf-8") as f:
                lines = f.readlines()
            text = ' '.join(lines)
            if folder == 'pos':
                label = 1
            else:
                label = 0
            data.append({
                'sentence': text,
                'label': label
            })
    return pd.DataFrame(data)

def identify_vn(df):
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    THRESHOLD = 0.9
    idx_not_vn = []
    for idx, row in df.iterrows():
        score = identifier.classify(row['sentence'])
        if score[0] != 'vi' or (score[0] == 'vi'and score[1] < THRESHOLD):
            idx_not_vn.append(idx)
    vn_df = df[~df.index.isin(idx_not_vn)]
    not_vn_df = df[df.index.isin(idx_not_vn)]
    return vn_df, not_vn_df

def clean_text(text):
    url_pattern = re.compile(r'https?://\s+\wwww\.\s+')
    text = url_pattern.sub(r" ", text)

    html_pattern = re.compile(r'<[^<>]+>')
    text = html_pattern.sub(" ", text)

    replace_chars = list(string.punctuation + string.digits)
    for char in replace_chars:
        text = text.replace(char, " ")
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U0001F600-\U0001F64F"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r" ", text)

    text = " ".join(text.split())
    return text.lower()

def preprocessing(df):
    df, _ = identify_vn(df)
    df['normalized_sentence'] = df['sentence'].apply(clean_text)
    return df