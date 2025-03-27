import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords

# ========== Preprocessing Functions ==========
def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["                   
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']  # Customize as needed
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

# ========== Dataset Class ==========
class TextDatasetTFIDF(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path).fillna('').sample(frac=1, random_state=42)

        # Merge Title and Body → new column 'text'
        df['Title+Body'] = df.apply(
            lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
            axis=1
        )

        # Rename columns as per preprocessing script
        df = df.rename(columns={
            "Unnamed: 0": "id",
            "class": "sentiment",
            "Title+Body": "text"
        })
        print("Preprocessed DataFrame:\n", df[['text', 'sentiment']].head())
        # Text cleaning
        df['text'] = df['text'].apply(remove_html)
        df['text'] = df['text'].apply(remove_emoji)
        df['text'] = df['text'].apply(remove_stopwords)
        df['text'] = df['text'].apply(clean_str)

       

        # # TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        self.X = torch.tensor(self.vectorizer.fit_transform(df['text']).toarray(), dtype=torch.float32)
        self.y = torch.tensor(df['sentiment'].values)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
