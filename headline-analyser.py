# AUTHOR:       Daniel Welsh
# CREATED:      15/01/2017
# DESCRIPTION:
#               A python script that pulls headlines from popular news sites using the
#               NewsAPI (http://www.newsapi.org) api and performs semantic analysis
#               with vaderSentiment to show the most positive and most negative
#               headlines at the current time.

import hashlib
import json
import urllib.request
from datetime import datetime
from json import JSONEncoder
import os
import time
import dotenv
import pickle
import string
import pandas as pd
import pymysql
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import urllib.parse

pymysql.install_as_MySQLdb()
import MySQLdb

s = set(stopwords.words('english'))


class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


class Headline(object):
    headline = ""
    link = ""
    semantic_value = 0.0
    origin = ""
    datetime = ""
    neg = 0.0
    pos = 0.0
    neu = 0.0
    predicted_class = 0
    display_image = ""

    def __str__(self, *args, **kwargs):
        return "{: <120} {: <20} {: <10} {: <25} {: <180} {: <25} {: <180}".format(self.headline, str(self.predicted_class),
                                                                          str(
                                                                              self.semantic_value), self.origin,
                                                                          self.link, str(self.datetime), self.display_image or "")

    def __hash__(self, *args, **kwargs):

        string = "{: <120} {: <10} {: <25} {: <180} {: <10} {: <10} {: <10}".format(self.headline,
                                                                                    str(
                                                                                        self.semantic_value),
                                                                                    self.origin, self.link,
                                                                                    str(self.pos), str(
                                                                                        self.neg),
                                                                                    str(self.neu))

        return hash(string)

    def sha256(self):
        string = "{: <120} {: <10} {: <25} {: <180} {: <10} {: <10} {: <10}".format(self.headline,
                                                                                    str(
                                                                                        self.semantic_value),
                                                                                    self.origin, self.link,
                                                                                    str(self.pos), str(
                                                                                        self.neg),
                                                                                    str(self.neu))

        hash_object = hashlib.sha256(bytes(string, 'utf-8'))
        hex_dig = hash_object.hexdigest()

        return hex_dig

    # FROM: http://stackoverflow.com/questions/3768895/how-to-make-a-class-json-serializable
    def default(self, o):
        try:
            iterable = iter(o)
        except TypeError:
            pass
        else:
            return list(iterable)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)

    def to_array(self):
        return [self.headline, self.origin, self.semantic_value, self.pos, self.neg, self.neu, self.datetime]

    def __init__(self, headline, link, origin, datetime, display_image):
        self.headline = headline
        self.link = link
        self.origin = origin
        self.datetime = datetime
        self.display_image = display_image


def connect_to_db():
    """
    Creates a connection to a database.

    :return: database, cursor
    """
    global db_host
    global db_user
    global db_password
    global database
    db = MySQLdb.connect(host=db_host, user=db_user,
                         passwd=db_password, db=database, charset='utf8')
    return db, db.cursor()


def load_classifier():
    """
    Loads the most recent classifier from file with the highest f1 score.

    :return: classifier, vectorizer
    """

    sql = "SELECT * FROM classifier_scores ORDER BY date_added DESC LIMIT 4"
    db, cur = connect_to_db()

    rows = pd.read_sql(sql, db)
    db.close()

    id = rows['f1'].idxmax()
    classifier_inf = rows.iloc[id].values

    fn = os.path.join(os.path.dirname(__file__),
                      'classifiers/' + classifier_inf[1] + '.clf')
    classifier = pickle.load(open(fn, 'rb'))
    fn = os.path.join(os.path.dirname(__file__), 'classifiers/vectorizer.vc')
    vectorizer = pickle.load(open(fn, 'rb'))

    return classifier, vectorizer


def analyze_headlines(blocks):
    """
    Performs semantic analysis on the headline and saves as Headline data-type

    :param blocks: array of Headlines
    :return: array of headlines, array items = [headline, link, origin, and date published]
    """
    analyzer = SentimentIntensityAnalyzer()
    headlines = []

    for block in blocks:
        vs = analyzer.polarity_scores(block.headline)

        block.semantic_value = vs['compound']
        block.pos = vs['pos']
        block.neg = vs['neg']
        block.neu = vs['neu']

        headlines.append(block)

    return headlines


def get_headlines(url):
    """
    Gets headlines from http://www.newsapi.org

    :param url: url
    :return: array of headlines, array items = [headline, link, origin, and date published]
    """
    headlines = []

    url += dotenv.get('NEWS_API_KEY', 'no-key')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req).read().decode('utf8')
    r = json.loads(response)

    prev_published_at = str(datetime.now()).split(" ")[0]

    for re in r['articles']:

        if str(re['publishedAt']) == 'None':
            published_at = prev_published_at
        else:
            published_at = "" + str(re['publishedAt']).split('T')[0]
            prev_published_at = published_at

        h_url = urllib.parse.urlparse(re['url'])
        u = h_url.scheme + '://' + h_url.netloc + h_url.path
        h = Headline(re['title'].split('\n')[0], u, re['source']['id'] or re['source']['name'], published_at, re['urlToImage'])
        headlines.append(h)

    return headlines


def save_to_db(headlines):
    """
    Save headlines to database

    :param headlines: all headlines
    :return: null
    """
    db = MySQLdb.connect(host=db_host, user=db_user, passwd=db_password, db=database,
                         charset='utf8')  # name of the data base

    cur = db.cursor()
    cur.execute('SET NAMES utf8;')
    cur.execute('SET CHARACTER SET utf8;')
    cur.execute('SET character_set_connection=utf8;')

    sql = "INSERT INTO headlines (headline, predicted_class, link, origin, semantic_value, hashcode, published_at, pos, neg, neu, display_image) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    for h in headlines:
        print("Added Headline:\t" + h.sha256())
        try:
            cur.execute(sql,
                        (h.headline, int(h.predicted_class), h.link, h.origin, h.semantic_value, h.sha256(), h.datetime,
                         h.pos, h.neg, h.neu, h.display_image))
        except pymysql.err.IntegrityError as e:
            print("ERROR: {}".format(e))
            continue
        db.commit()

    db.close()


def print_to_file(headlines):
    """
    Write all headlines to file.

    :param headlines: all headlines
    :return: null
    """
    f = open('headlines.txt', 'w')

    f.write("{:-<359}".format('-') + '\n')
    f.write("{: <120} {: <10} {: <25} {: <180} {: <25}".format('Headline', 'Semantic', 'Origin', 'Link',
                                                               'Date-Time') + '\n')
    f.write("{:-<359}".format('-') + '\n')

    i = 0
    idx = 0
    l_idx = 0
    highest = 0.0
    lowest = 0.0
    cumulative_sentiment = 0.0
    for h in headlines:
        f.write(str(h) + '\n')

        if float(h.semantic_value) > highest:
            highest = float(h.semantic_value)
            idx = i

        if lowest > float(h.semantic_value):
            lowest = float(h.semantic_value)
            l_idx = i

        cumulative_sentiment += float(h.semantic_value)

        i += 1

    f.write('\n')

    f.write("{:-<359}".format('-') + '\n')

    f.write('Total headlines analysed: ' + str(len(headlines)) + '\n')
    f.write('Cumulative Sentiment:     ' + str(cumulative_sentiment) + '\n')

    f.write('\nThe most positive article of the day is:' + '\n')
    f.write(str(headlines[idx]) + '\n')

    f.write('\nThe most negative article of the day is:' + '\n')
    f.write(str(headlines[l_idx]) + '\n')


def print_results(headlines):
    """
    Prints and formats headlines

    :param headlines: Array of headlines
    :return: null
    """
    print("{:-<359}".format('-'))
    print(
        "{: <120} {: <20} {: <10} {: <25} {: <180} {: <25}".format('Headline', 'Predicted Class', 'Semantic', 'Origin',
                                                                   'Link', 'Date-Time'))
    print("{:-<359}".format('-'))

    i = 0
    idx = 0
    l_idx = 0
    highest = 0.0
    lowest = 0.0
    cumulative_sentiment = 0.0
    for h in headlines:
        print(str(h))

        if float(h.semantic_value) > highest:
            highest = float(h.semantic_value)
            idx = i

        if lowest > float(h.semantic_value):
            lowest = float(h.semantic_value)
            l_idx = i

        cumulative_sentiment += float(h.semantic_value)

        i += 1

    print('\n')

    print("{:-<359}".format('-'))

    print('Total headlines analysed: ' + str(len(headlines)))
    print('Cumulative Sentiment:     ' + str(cumulative_sentiment))

    print('\nThe most positive article of the day is:')
    print(str(headlines[idx]))

    print('\nThe most negative article of the day is:')
    print(str(headlines[l_idx]))


def to_epoch(date_time):
    """
    Convert datetime to epoch

    :param date_time: datetime
    :return: epoch
    """
    pattern = '%Y-%m-%d'
    epoch = int(time.mktime(time.strptime(str(date_time), pattern)))
    return epoch


def normalise_column(df, column):
    """
    Normalise a column of floats

    :param column: column name
    :return: dataframe with update column
    """

    max_c = df[column].max()
    min_c = df[column].min()

    if max_c == min_c:
        df[column] = df[column].apply(lambda x: 0)
    else:
        df[column] = df[column].apply(lambda x: float((x - min_c)) / float((max_c - min_c)))

    return df


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


def predict_class(all_headlines):
    """
    Predict whether each headline is negative or positive.

    :param all_headlines: all headlines
    :return: headlines with predictions
    """

    clf, v = load_classifier()

    headlines = []
    for h in all_headlines:
        headlines.append(h.to_array())

    df = pd.DataFrame(headlines)
    df.columns = \
        [
            'headline',
            'origin',
            'semantic_value',
            'pos',
            'neg',
            'neu',
            'published_at'
        ]

    df['headline'] = df['headline'].map(lambda x: strip_punctuation(x))
    df['headline'] = df['headline'].map(lambda x: x.lower())
    df['headline'] = df['headline'].map(lambda x: filter_stop_words(x))
    df['published_at'] = df['published_at'].map(lambda x: to_epoch(x))

    df = normalise_column(df, 'semantic_value')
    df = normalise_column(df, 'pos')
    df = normalise_column(df, 'neu')
    df = normalise_column(df, 'published_at')

    tr_counts = v.transform(df['headline'])

    tr = pd.DataFrame(tr_counts.todense())
    df.join(tr)

    output = clf.predict(df.drop(["headline", "origin"], axis=1)).astype(int)
    df['predicted_class'] = output

    i = 0
    for h in all_headlines:
        h.predicted_class = df['predicted_class'].loc[i]
        i += 1

    print(all_headlines)

    return all_headlines


raw_headlines = []
fn = os.path.join(os.path.dirname(__file__), '.env')
dotenv.load(fn)

key = dotenv.get('NEWS_API_KEY', 'no key')
database = dotenv.get('DATABASE', 'no database')
db_user = dotenv.get('DB_USER', 'no user')
db_password = dotenv.get('DB_PASSWORD', 'no password')
db_host = dotenv.get('DB_HOST', 'no host')

urls = \
    [
        'https://newsapi.org/v2/top-headlines?country=gb&apiKey=',
        'https://newsapi.org/v2/top-headlines?country=us&apiKey='
    ]

for url in urls:
    lines = get_headlines(url)
    raw_headlines.extend(lines)

all_headlines = analyze_headlines(raw_headlines)

all_headlines = predict_class(all_headlines)

print_results(all_headlines)
save_to_db(all_headlines)
