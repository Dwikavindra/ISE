########## 1. Import required libraries ##########

import pandas as pd
import numpy as np
import re
import math

# Text and feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)

# Classifier
from sklearn.naive_bayes import GaussianNB

# Text cleaning & stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

########## 2. Define text preprocessing methods ##########

def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Stopwords
NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']  # You can customize this list as needed
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

def remove_stopwords(text):
    """Remove stopwords from the text."""
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])


def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
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

def prepare_dataset(dataset):
    # Make sure output folder exists
    output_dir = 'title_body'
    os.makedirs(output_dir, exist_ok=True)

    # Define paths
    input_path = f'datasets/{dataset}.csv'
    output_path = os.path.join(output_dir, f'Title+Body_{dataset}.csv')

    # Load and shuffle data
    pd_all = pd.read_csv(input_path)
    pd_all = pd_all.sample(frac=1, random_state=999)

    # Merge Title and Body into one column
    pd_all['Title+Body'] = pd_all.apply(
        lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
        axis=1
    )

    # Rename and keep necessary columns
    pd_tplusb = pd_all.rename(columns={
        "Unnamed: 0": "id",
        "class": "sentiment",
        "Title+Body": "text"
    })

    # Save to new file
    pd_tplusb.to_csv(output_path, index=False, columns=["id", "Number", "sentiment", "text"])

    # Return the DataFrame
    return pd.read_csv(output_path).fillna('')

def check_and_clear_existing_file(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        print(f"⚠️  The file '{file_path}' already exists and is not empty.")
        response = input("Do you want to delete and start over? Type 'yes' to confirm: ").strip().lower()
        if response != 'yes':
            print("❌ Operation cancelled by user. No changes made.")
            exit(1)
        else:
            os.remove(file_path)
            print(f"✅ File '{file_path}' has been deleted. Starting fresh.")
########## 3. Download & read data ##########
import os
import subprocess
# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
project = 'tensorflow'
path = f'datasets/{project}.csv'

pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)  # Shuffle

# Merge Title and Body into a single column; if Body is NaN, use Title only
pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)

# Keep only necessary columns: id, Number, sentiment, text (merged Title+Body)
pd_tplusb = pd_all.rename(columns={
    "Unnamed: 0": "id",
    "class": "sentiment",
    "Title+Body": "text"
})
pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])

def write_row_to_csv(file_path, columns, values):

    assert len(columns) == len(values), "Columns and values must be the same length."
    folder = os.path.dirname(file_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    df = pd.DataFrame([dict(zip(columns, values))])
    df.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))
########## 4. Configure parameters & Start training ##########

# ========== Key Configurations ==========

# 1) Data file to read
datafile = 'Title+Body.csv'

# 2) Number of repeated experiments
REPEAT = 20

# 3) Output CSV file name
out_csv_name = f'../{project}_NB.csv'

# ========== Read and clean data ==========
data = pd.read_csv(datafile).fillna('')
text_col = 'text'

# Keep a copy for referencing original data if needed
original_data = data.copy()

# Text cleaning
data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(remove_stopwords)
data[text_col] = data[text_col].apply(clean_str)

# ========== Hyperparameter grid ==========
# We use logspace for var_smoothing: [1e-12, 1e-11, ..., 1]
params = {
    'var_smoothing': np.logspace(-12, 0, 13)
}



# Lists to store metrics across repeated runs
accuracies  = []
precisions  = []
recalls     = []
f1_scores   = []
auc_values  = []
final_clf=None
for repeated_time in range(REPEAT):
    indices = np.arange(data.shape[0])
    train_index, test_index = train_test_split(
        indices, test_size=0.2, random_state=repeated_time
    )
    train_text = data[text_col].iloc[train_index]
    test_text = data[text_col].iloc[test_index]

    y_train = data['sentiment'].iloc[train_index]
    y_test  = data['sentiment'].iloc[test_index]

    # --- 4.2 TF-IDF vectorization ---
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1000  # Adjust as needed
    )
    X_train = tfidf.fit_transform(train_text)
    X_test = tfidf.transform(test_text)
   
    # --- 4.3 Naive Bayes model & GridSearch ---
    clf = GaussianNB()
    grid = GridSearchCV(
        clf,
        params,
        cv=5,              # 5-fold CV (can be changed)
        scoring='roc_auc'  # Using roc_auc as the metric for selection
    )
    grid.fit(X_train.toarray(), y_train)

    # Retrieve the best model
    best_clf = grid.best_estimator_
    best_clf.fit(X_train.toarray(), y_train)

    # --- 4.4 Make predictions & evaluate ---
    y_pred = best_clf.predict(X_test.toarray())

    if(repeated_time== REPEAT-1):
        final_clf=best_clf 

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    prec = precision_score(y_test, y_pred, average='macro')
    precisions.append(acc)
    rec = recall_score(y_test, y_pred, average='macro')
    recalls.append(rec)
    f1 = f1_score(y_test, y_pred, average='macro')
    f1_scores.append(f1)
    fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
    auc_val = auc(fpr, tpr)
    auc_values.append(auc_val)
    file_path=f'baselines_data/{project}_NB_detailed_baseline.csv'
    if(repeated_time==0):
       check_and_clear_existing_file(file_path)
    write_row_to_csv(
        file_path=file_path,
        columns=['iteration', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
        values=[repeated_time, acc, prec, rec, f1, auc_val]
    )

final_accuracy  = np.mean(accuracies)
final_precision = np.mean(precisions)
final_recall    = np.mean(recalls)
final_f1        = np.mean(f1_scores)
final_auc       = np.mean(auc_values)

print("=== Naive Bayes + TF-IDF Results ===")
print(f"Number of repeats:     {REPEAT}")
print(f"Average Accuracy:      {final_accuracy:.4f}")
print(f"Average Precision:     {final_precision:.4f}")
print(f"Average Recall:        {final_recall:.4f}")
print(f"Average F1 score:      {final_f1:.4f}")
print(f"Average AUC:           {final_auc:.4f}")

DATASETS=["caffe","pytorch","keras","incubator-mxnet"]
for project in DATASETS:
    data=prepare_dataset(project)
    for i in range (REPEAT):
        data = data.sample(frac=1).reset_index(drop=True)
        texts = data['text']
        labels = data['sentiment']
        X_all = tfidf.transform(texts)  
        y_pred = final_clf.predict(X_all.toarray())
        acc = accuracy_score(labels, y_pred)
        prec = precision_score(labels, y_pred, average='macro')
        rec = recall_score(labels, y_pred, average='macro')
        f1 = f1_score(labels, y_pred, average='macro')
        fpr, tpr, _ = roc_curve(labels, y_pred, pos_label=1)
        auc_val = auc(fpr, tpr)
        file_path=f'baselines_data/{project}_NB_tensorflow_based_detailed_baseline.csv'
        if(i==0):
            check_and_clear_existing_file(file_path)
        write_row_to_csv(
            file_path=file_path,
            columns=['iteration', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
            values=[i, acc, prec, rec, f1, auc_val]
        )



