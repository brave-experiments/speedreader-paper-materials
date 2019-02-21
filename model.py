from __future__ import print_function
import string, argparse, codecs, json, re, sqlite3
from multiprocessing import Pool
from collections import Counter, OrderedDict
from bs4 import BeautifulSoup, Tag
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split

#logging and pretty printing
from tqdm import tqdm
from tabulate import tabulate


def read_data(db_name):
    """Read raw data from database and return a DataFrame 
    corresponding to the result. 
    """
    try:
        conn = sqlite3.connect(db_name)
	raw_data = pd.read_sql_query("SELECT * FROM speedReader;", conn)
	conn.close()
        return raw_data
    except Exception as e:
        print(e)

def preprocess(row):
    """For each DataFrame row as an input, parse the HTML, 
       extract and return features.
    """
    soup = BeautifulSoup(row['html'], 'html.parser')
    features = extract_features(soup, url=row['url'])
    features['class'] = row['class']
    return features
		
def prepare(data):
    """Return input samples (X) and corresponding outputs (y) necessary 
    for training the model
    """
    data = pd.DataFrame(data)
    X = data.drop('class', axis=1)
    y = data['class']
    return X, y

def extract_features(soup, url):
    """Extract features from BeautifulSoup object and return 
    features as an ordered dictionary. 
    """
    features = OrderedDict()
    min_num_chars = 400
    high_score_tag_counter, high_score_words_counter, high_score_text_block = list(), 0, 0

    for tag in soup.descendants:
            if isinstance(tag, Tag):
                    if tag.name and tag.name.upper() in ["BLOCKQUOTE", "DL", "DIV", "OL", "P", "PRE", "TABLE", "UL", "SELECT", "ARTICLE", "SECTION"]:
                            high_score_tag_counter.append(tag.name.lower())
                            text = tag.find(text=True, recursive=False)
                            if text and len("".join(text.split())) >= min_num_chars:
                                    high_score_text_block += 1
                                    text = text.translate({ord(k): None for k in string.punctuation})
                                    high_score_words_counter += len(text.split())

    features['images'] = len(soup.find_all('img'))
    features['anchors'] = len(soup.find_all('a'))
    features['scripts'] = len(soup.find_all('script'))
    features['text_blocks'] = high_score_text_block
    features['words'] = high_score_words_counter
    high_score_tag_counter = Counter(high_score_tag_counter)
    for _ in ["BLOCKQUOTE", "DL", "DIV", "OL", "P", "PRE", "TABLE", "UL", "SELECT", "ARTICLE", "SECTION"]:
            if _.lower() in high_score_tag_counter:
                    features[_.lower()] = high_score_tag_counter[_.lower()]
            else:
                    features[_.lower()] = 0

    if url:
            features['url_depth'] = len(url.split('://')[1].split('/')[1:])
    features['amphtml'] = 1 if soup.find('link', rel='amphtml') else 0
    features['fb_pages'] = 1 if soup.find('meta', property='fb:pages') else 0
    features['og_article'] = 1 if soup.find('meta', attrs={'property': 'og:type', 'content': 'article'}) else 0
    schemaOrgRegex = re.compile('http(s)?:\/\/schema.org\/(Article|NewsArticle|APIReference)')
    if schemaOrgRegex.search(soup.text):
            features['schema_org_article'] = 1
    else:
            features['schema_org_article'] = 0

    return features

def classifier(X, y):
    """Train the classifier using the inputs (X) and outputs (y), 
    and return the model.
    """
    print("[+] Training the model")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=10)
    model = RandomForestClassifier(n_estimators=50, random_state=10, n_jobs=4, oob_score=True)
    model.fit(X_train, y_train)

    return model

def cross_validation(model, X, y):
    """Do 10 fold cross validation and print final evaluation results."""

    print("[+] Preparing classification report")
    score = cross_validate(model, X=X, y=y, scoring=['precision', 'recall', 'f1', 'accuracy'], cv=10, return_train_score=False)
    print(tabulate([["Precision","%0.2f (+/- %0.2f)" % (np.mean(score['test_precision']), np.std(score['test_precision']) * 2)], 
        ["Recall","%0.2f (+/- %0.2f)" % (np.mean(score['test_recall']), np.std(score['test_precision']) * 2)],
        ["F-1","%0.2f (+/- %0.2f)" % (np.mean(score['test_f1']), np.std(score['test_f1']) * 2)],
        ["Accuracy", "%0.2f (+/- %0.2f)" % (np.mean(score['test_accuracy']), np.std(score['test_accuracy']) * 2)]], ["Metric", "Value"], tablefmt="grid"))

def run(dbname, threads):
    """The core function: read raw data from database and extract features 
    from each row in a separate process. After preparing data, train the model
    and evaluate.
    """
    pool = Pool(processes=threads)
    raw_data = read_data(dbname)
    data = pool.map_async(preprocess, tqdm([row for name, row in raw_data.iterrows()], ncols=80, desc="[+] Preprocessing data: "))
    data.wait()
    data = data.get()
    X, y = prepare(data)
    model = classifier(X, y)
    cross_validation(model, X, y)
    if pool:
        pool.close()
        pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbname', help="Database filename", required=True)
    parser.add_argument('--threads', help="Number of threads to run preprocessing", type=int, default=1)

    args = parser.parse_args()

    run(args.dbname, args.threads)
	
        







