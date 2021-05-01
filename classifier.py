# DESCRIPTION:
# - Script to train a classifier from annotated headline data, to then predict sentiment of future classifiers.

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import time
import pandas as pd
import pickle
import os
from common import connect_to_db, load_file, filter_stop_words, strip_punctuation, normalise_column
from datetime import datetime

c_name = ['SVM', 'RFR', 'KNN']


def fetch_data_set():
    """
    Gets all annotated headlines from the server.

    :return: pandas data frame of annotated headline data.
    """

    sql = """
        SELECT h.id as headline_id,
               headline,
               origin,
               semantic_value,
               pos,
               neg,
               neu,
               published_at,
               votes.positive,
               votes.negative
        FROM headlines as h
        JOIN (
            SELECT sum(a.positive) AS positive,
                   sum(a.negative) AS negative,
                   a.headline_id
            FROM annotations as a
            GROUP BY a.headline_id
        ) as votes
           ON h.id = votes.headline_id
    """
    db = connect_to_db()
    cur = db.cursor()
    cur.execute(sql)

    df = pd.read_sql(sql, con=db)
    db.close()

    return df


def engineer_data_set(df):
    # Remove all rows where headline id is null.
    # Values are set to null when the auto increment value is
    # missing, some bug with pandas and postgres?
    df = df.dropna(subset=['headline_id'])
    df = df.drop(['headline_id'], axis=1)

    # Get ground truth
    df['negative'] = df['negative'].map(lambda x: x * -1)
    df = df.assign(truth=lambda x: x['positive'] + x['negative'])

    # Marks split annotations as negative, could have a third class.
    df['truth'] = df['truth'].map(lambda x: 1 if x > 0 else 0)

    df['headline'] = df['headline'].map(lambda x: strip_punctuation(x))
    df['headline'] = df['headline'].map(lambda x: x.lower())
    df['headline'] = df['headline'].map(lambda x: filter_stop_words(x))

    df['published_at'] = df['published_at'].map(lambda x: to_epoch(x))

    df = normalise_column(df, 'published_at')
    df = df.drop(['positive', 'negative'], axis=1)

    return df


def to_epoch(date_time):
    """
    Convert datetime to epoch

    :param date_time: datetime
    :return: epoch
    """
    pattern = '%Y-%m-%d'
    earliest = datetime(1970, 1, 1, 0, 0, 0, 0)

    if date_time is None:
        date_time = str(datetime.now()).split(" ")[0]

    if time.strptime(str(date_time), pattern).tm_year < earliest.year:
        date_time = str(earliest).split(" ")[0]

    epoch = int(time.mktime(time.strptime(str(date_time), pattern)))
    return epoch


def create_classifiers():
    """
    Initialise classifiers:
        SVC: Support vector machine
        MLP: Multi layered perception
        RFR: Random Forest

    :return: Array of classifiers.
    """
    classifiers = \
        [
            SVC(C=512, verbose=False, kernel='rbf', probability=True),
            RandomForestClassifier(
                n_estimators=160, max_features='sqrt', verbose=False),
            KNeighborsClassifier(
                n_neighbors=170, algorithm='auto', metric='euclidean', weights='uniform')
        ]

    return classifiers


def train_classifiers(data, classifiers):
    """
    Train classifiers

    :param classifiers: Array of classifiers
    :param data: data set
    :return: trained classifiers
    """
    count_vectorizer = CountVectorizer()
    tr_counts = count_vectorizer.fit_transform(data['headline'])

    tr = pd.DataFrame(tr_counts.todense())
    data.join(tr)

    for clf in classifiers:
        clf = clf.fit(data.drop(
            ["headline", "origin", "truth"], axis=1).values, data['truth'].values)

    return count_vectorizer, classifiers


# Make a prediction on the test set
def make_prediction(vectorizer, classifiers, data):
    """
    Use classifiers to make a prediction.

    :param classifiers: array of classifiers
    :param data: headline to predict
    :return: list of classifiers
    """

    tr_counts = vectorizer.transform(data['headline'])

    tr = pd.DataFrame(tr_counts.todense())
    data.join(tr)

    for clf in classifiers:
        output = clf.predict(
            data.drop(["headline", "origin", "truth"], axis=1)).astype(int)

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
                'C': [
                    0.1,
                    0.5,
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    64,
                    128,
                    256,
                    512
                ],
                'kernel':
                    [
                        'linear',
                        'poly',
                        'rbf'
                    ]
            },
            # {
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
            # },
            {
                'n_estimators': [
                    110, 120, 130, 140, 150, 160, 170, 180, 190,
                ],
            },
            {
                'n_neighbors':
                    [
                        10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                        110, 120, 130, 140, 150, 160, 170, 180, 190,
                        200, 210, 220, 230, 240, 250
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
        grid = GridSearchCV(
            classifiers[i], param_grid=ps[i], cv=5, verbose=3, scoring=f1_scorer)
        grid.fit(train.drop(["headline", "origin", "truth"], axis=1).values, train['truth'].values)
        scores = grid.best_score_
        best_parameters = grid.best_estimator_.get_params()
        param_list = ''
        for param_name in sorted(ps[i].keys()):
            param_list += '\t%s: %r\n' % (param_name,
                                          best_parameters[param_name])

        b_params[i] = '%s\nBest score: %0.3f \nBest parameters set: %s' % (
            scores, grid.best_score_, param_list)

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

    db = connect_to_db()

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

            clf = clf.fit(train_df.drop(
                ["headline", "origin", "truth"], axis=1).values, train_df['truth'].values)
            output = clf.predict(test_df.drop(
                ["headline", "origin", "truth"], axis=1).values).astype(int)

            accuracy = accuracy_score(output, df['truth'].iloc[test].values)
            f1 = f1_score(output, df['truth'].iloc[test].values)
            acc = np.append(acc, accuracy)
            f1s = np.append(f1s, f1)
            confusion += confusion_matrix(
                df['truth'].iloc[test].values, output)

        score_string = "{: <25} {: <25} {: <25} {: <25} {: <25}".format(c_name[i], acc.mean(), acc.std(), f1s.mean(),
                                                                        f1s.std())

        sql = """
            INSERT INTO classifier_scores 
                (name, accuracy, accuracy_std, f1, f1_std, data_size) 
            VALUES 
                (%s, %s, %s, %s, %s, %s)
        """

        cursor = db.cursor()
        cursor.execute(sql, (c_name[i], float(acc.mean()), float(
            acc.std()), float(f1s.mean()), float(f1s.std()), len(df)))
        db.commit()

        print(score_string)
        print(confusion)

        i += 1

    db.close()


def save_classifiers(classifiers, vectorizer):
    """
    Pickle classifiers
    :type classifiers: list
    :param classifiers: Classifiers to save.
    :param vectorizer: Vectorizer to save.
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

    classifiers = []
    for classifier_name in c_name:
        classifier = load_file(
            'classifiers/{classifier}.clf'.format(classifier=classifier_name))
        classifiers.append(classifier)

    return classifiers


def load_vectorizer():
    """
    Get the vectorizer from file.

    :return: vectorizer
    """

    return load_file('classifiers/vectorizer.vc')


def main():
    df = fetch_data_set()
    df = engineer_data_set(df)

    print("Collected {count} annotated headline(s)".format(count=len(df)))

    classifiers = create_classifiers()

    # Uncomment to optimise parameters of the classifiers
    # optimise_parameters(classifiers, df)

    vectorizer, classifiers = train_classifiers(df, classifiers)
    cross_val_classifiers(classifiers, df)

    save_classifiers(classifiers, vectorizer)

    # classifiers = load_classifiers()
    # v = load_vectorizer()
    #
    # make_prediction(v, df)


if __name__ == "__main__":
    main()
