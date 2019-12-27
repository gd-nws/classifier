# AUTHOR: Daniel
# Welsh, d.welsh @ ncl.ac.uk
# DATE: 27 / 02 / 2017
# DESCRIPTION:
# - Script to train a classifier from annotated headline data, to then predict sentiment of future classifiers.

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import csv
import numpy as np
import dotenv
import time
import pymysql
import pandas as pd
import string
import pickle
from nltk.corpus import stopwords
import os

# import warnings
# warnings.filterwarnings("ignore")

pymysql.install_as_MySQLdb()
import MySQLdb

s = set(stopwords.words('english'))
c_name = ['SVM', 'MLP', 'RFR', 'KNN']

# Get data from DB...

# Pre-process data.
#     Strip common words
#     Limit to 20 words - 6 letters each - 120 characters
#     Convert to ascii string
#     Normalize sentiment scores
#     Map for origins
#     Date to normalized integers

# Train classifiers
#     Grid search SVM params

# Compare parameters

# Save best model

# ***********************************************************************************************************************
# GET DATA
# ***********************************************************************************************************************

fn = os.path.join(os.path.dirname(__file__), '.env')
dotenv.load(fn)
database = dotenv.get('DATABASE', 'no database')
db_user = dotenv.get('DB_USER', 'no user')
db_password = dotenv.get('DB_PASSWORD', 'no password')
db_host = dotenv.get('DB_HOST', 'no host')


def get_data(db, cur):
    """
    Gets all annotated headlines from the server.

    :param db: database connection
    :param cur: cursor
    :return: pandas dataframe of annotated headline data.
    """
    cur.execute('SET NAMES utf8;')
    cur.execute('SET CHARACTER SET utf8;')
    cur.execute('SET character_set_connection=utf8;')

    sql = """
        SELECT annotations.id, headline, origin, semantic_value, pos, neg, neu, published_at, positive, negative
        FROM annotations
        LEFT JOIN headlines
        ON headlines.id = annotations.id
    """
    cur.execute(sql)

    df = pd.read_sql(sql, con=db)
    db.close()

    # Get ground truth
    df['negative'] = df['negative'].map(lambda x: x * -1)
    df = df.assign(truth=lambda x: x['positive'] + x['negative'])

    # Marks split annotations as negative, could have a third class.
    df['truth'] = df['truth'].map(lambda x: 1 if x > 0 else 0)

    return df


def connect_to_db():
    """
    Connect to a database.

    :return: database and cursor
    """
    global db_host
    global db_user
    global db_password
    global database
    db = MySQLdb.connect(host=db_host, user=db_user, passwd=db_password, db=database, charset='utf8')
    return db, db.cursor()


def filter_stop_words(headline):
    """
    Filter all stop words from a string to reduce headline size.

    :param headline: full headline
    :return: shortened headline
    """
    words = filter(lambda w: not w in s, headline.split())
    line = ""
    l = 0
    for w in words:
        if l < 20:
            line += w + " "
            l += 1
        else:
            return line.strip()
    return line.strip()


def strip_punctuation(headline):
    """
    Remove all punctuation from a headline to return only the words.

    :param headline: raw headline
    :return: parsed headline
    """
    table = headline.maketrans({key: None for key in string.punctuation})
    headline = headline.translate(table)
    return headline


def convert_to_ascii(word):
    """
    Convert a string into ascii representation
    SOURCE: http://stackoverflow.com/questions/8452961/convert-string-to-ascii-value-python

    :param word: Plaintext word
    :return: Ascii representation
    """

    if word is None:
        word = " "

    word = ''.join(str(ord(c)) for c in word)
    return word


def to_epoch(date_time):
    """
    Convert datetime to epoch

    :param date_time: datetime
    :return: epoch
    """
    pattern = '%Y-%m-%d'
    epoch = int(time.mktime(time.strptime(str(date_time), pattern)))
    return epoch


def normalise_column(column):
    """
    Normalise a column of floats

    :param column: column name
    :return: dataframe with update column
    """
    global df
    max_c = df[column].max()
    min_c = df[column].min()
    df[column] = df[column].apply(lambda x: (x - min_c) / (max_c - min_c))

    return df


# Write predictions to file.
def log_predictions(df, output, file_name):
    """
    log predictions to file.

    :param output: predictions
    :param ids: id of predicted headline
    :param file_name: file name
    :return: null
    """
    fn = os.path.join(os.path.dirname(__file__), './predictions/')
    predictions_file = open(fn + file_name, "w")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["ID", "Class", "Truth", "Headline"])
    open_file_object.writerows(zip(df['id'], output, df["truth"], df['headline']))
    predictions_file.close()


def init_classifiers():
    """
    Initialise classifiers:
        SVC: Support vector machine
        MLP: Multi layered perception
        RFR: Random Forest

    :return: Array of classifiers.
    """
    classifiers = \
        [
            SVC(C=0.1, verbose=False, kernel='rbf', probability=True),
            RandomForestClassifier(n_estimators=20, max_features='sqrt', verbose=False),
            KNeighborsClassifier(n_neighbors=20, algorithm='auto', metric='euclidean', weights='uniform')
        ]

    return classifiers


def train_classifiers(df):
    """
    Train classifiers

    :param classifiers: Array of classifiers
    :param train: train data
    :param truth: truth data
    :return: trained classifiers
    """
    global classifiers

    count_vectorizer = CountVectorizer()
    tr_counts = count_vectorizer.fit_transform(df['headline'])

    tr = pd.DataFrame(tr_counts.todense())
    df.join(tr)

    for clf in classifiers:
        clf = clf.fit(df.drop(["headline", "origin", "truth", "id"], axis=1).values, df['truth'].values)

    return count_vectorizer


# Make a prediction on the test set
def make_prediction(vectorizer, data):
    """
    Use classifiers to make a prediction.

    :param classifiers: array of classifiers
    :param data: headline to predict
    :return: list of classifiers
    """

    tr_counts = vectorizer.transform(df['headline'])

    tr = pd.DataFrame(tr_counts.todense())
    df.join(tr)

    i = 0
    # print test

    for clf in classifiers:
        output = clf.predict(df.drop(["headline", "origin", "truth", "id"], axis=1)).astype(int)
        # log_predictions(data, output, c_name[i] + "_predictions.csv")

        i += 1
    return classifiers


def optimise_parameters(classifiers, train):
    """
    Grid search optimum parameters for classifiers.

    :param classifiers: Array of classifiers.
    :param train: Training data.
    :return: null
    """

    ps = \
        [
            {
                # 'C': np.arange(15, 30, 0.5),
                'C': [0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                'kernel':
                    [
                        # 'linear',
                        # 'poly',
                        'rbf'
                    ]
            },
            {
                # 'solver': ["lbfgs", "sgd", "adam"],
                # "learning_rate": ["constant", "invscaling", "adaptive"],
                # "activation": ["identity", "logistic", 'tanh', "relu"],
                # "hidden_layer_sizes": [
                #     (500, 250, 100, 10),
                #     (600, 400, 200, 100, 50, 10),
                #     (8, 5, 2),
                #     (50, 20, 10, 2),
                #     (100, 50, 20, 10, 5, 2),
                #     (10, 10, 10, 10, 10, 10, 10, 10, 10, 10)
                # ]
            },
            {
                'n_estimators': [
                    10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                    110, 120, 130, 140, 150, 160, 170, 180, 190,
                    200, 210, 220, 230, 240, 250
                ],
                'max_features': ['auto', 'sqrt', 'log2', None]
            },
            {
                'n_neighbors':
                    [
                        3,
                        5,
                        8,
                        10,
                        20
                    ],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'metric': ['euclidean', 'minkowski', 'manhattan']
            }
        ]

    i = 0

    b_params = ['', '', '', '']
    f1_scorer = make_scorer(f1_score, pos_label=1)

    print(train.drop(["headline", "origin", "truth"], axis=1))

    while i < len(classifiers):
        grid = GridSearchCV(classifiers[i], param_grid=ps[i], cv=5, verbose=3, scoring=f1_scorer)
        grid.fit(train.drop(["headline", "origin", "truth", "id"], axis=1).values, train['truth'].values)
        scores = grid.grid_scores_
        best_parameters = grid.best_estimator_.get_params()
        param_list = ''
        for param_name in sorted(ps[i].keys()):
            param_list += '\t%s: %r\n' % (param_name, best_parameters[param_name])

        b_params[i] = '%s\nBest score: %0.3f \nBest parameters set: %s' % (scores, grid.best_score_, param_list)

        i += 1

    for pars in b_params:
        print(pars)


# Cross validate classifiers
def cross_val_classifiers(classifiers, df):
    """
    Cross validate classifiers using a stratified K-Fold

    :param classifiers: Array fo classifiers
    :param df: data
    :return: null
    """

    skf = StratifiedKFold(n_splits=5)
    acc = np.empty(0)
    f1s = np.empty(0)

    i = 0

    # Use a K-Folding technique to generate average accuracy and F1 scores.
    score_header = "{: <25} {: <25} {: <25} {: <25} {: <25}".format("Classifier Name", "Average Accuracy",
                                                                    "Accuracy STD", "Average F1", "F1 STD")
    print(score_header)

    db, cur = connect_to_db()

    for clf in classifiers:
        confusion = np.array([[0, 0], [0, 0]])
        for train, test in skf.split(df['headline'], df['truth']):
            train_df = df.loc[train]
            test_df = df.loc[test]

            count_vectorizer = CountVectorizer()
            tr_counts = count_vectorizer.fit_transform(train_df['headline'])
            te_counts = count_vectorizer.transform(test_df['headline'])

            tr = pd.DataFrame(tr_counts.todense())
            train_df.join(tr)

            te = pd.DataFrame(te_counts.todense())
            test_df.join(te)

            clf = clf.fit(train_df.drop(["headline", "origin", "truth", "id"], axis=1).values, train_df['truth'].values)
            output = clf.predict(test_df.drop(["headline", "origin", "truth", "id"], axis=1).values).astype(int)

            accuracy = accuracy_score(output, df['truth'].iloc[test].values)
            f1 = f1_score(output, df['truth'].iloc[test].values)
            acc = np.append(acc, accuracy)
            f1s = np.append(f1s, f1)
            confusion += confusion_matrix(df['truth'].iloc[test].values, output)

        score_string = "{: <25} {: <25} {: <25} {: <25} {: <25}".format(c_name[i], acc.mean(), acc.std(), f1s.mean(),
                                                                        f1s.std())

        sql = "INSERT INTO classifier_scores (name, accuracy, accuracy_std, f1, f1_std, data_size) VALUES (%s, %s, %s, %s, %s, %s)"

        cur.execute(sql, (c_name[i], float(acc.mean()), float(acc.std()), float(f1s.mean()), float(f1s.std()), len(df)))
        db.commit()

        print(score_string)

        print(confusion)

        i += 1

    db.close()


def save_classifiers(classifiers, vectorizer):
    """
    Pickle classifiers
    :return: null
    """

    i = 0
    fn = os.path.join(os.path.dirname(__file__), './classifiers/')
    for clf in classifiers:
        pickle.dump(clf, open(fn + c_name[i] + ".clf", "wb"))
        i += 1

    pickle.dump(vectorizer, open(fn + "vectorizer.vc", "wb"))


def load_classifiers():
    """
    Load classifiers from file.
    :return: classifiers
    """

    fn = os.path.join(os.path.dirname(__file__), './classifiers/')
    clfs = []
    for c in c_name:
        clf = pickle.load(open(fn + c + ".clf", "rb"))
        clfs.append(clf)

    return clfs


def load_vectorizer():
    """
    Get the vectorizer from file.

    :return: vectorizer
    """

    fn = os.path.join(os.path.dirname(__file__), './classifiers/')
    return pickle.load(open(fn + "vectorizer.vc", "rb"))


db, cur = connect_to_db()
df = get_data(db, cur)

df['headline'] = df['headline'].map(lambda x: strip_punctuation(x))
df['headline'] = df['headline'].map(lambda x: x.lower())
df['headline'] = df['headline'].map(lambda x: filter_stop_words(x))

df['published_at'] = df['published_at'].map(lambda x: to_epoch(x))

df = normalise_column('published_at')
df = df.drop(['positive', 'negative'], axis=1)

print(len(df))

classifiers = init_classifiers()

# Uncomment to optimise parameters of the classifiers
# optimise_parameters(classifiers, df)

v = train_classifiers(df)
cross_val_classifiers(classifiers, df)


save_classifiers(classifiers, v)

# classifiers = load_classifiers()
# v = load_vectorizer()
#
# make_prediction(v, df)
