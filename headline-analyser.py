# DESCRIPTION:
#               Pull headlines from NewsAPI, and classify them as positive or negative.
#               Headlines can be printed or persited.

from common import connect_to_db, normalise_column, to_epoch
from headline import Headline

import sys
import hashlib
import json
import urllib.request
from json import JSONEncoder
import os
import time
from datetime import datetime
import pickle
import string
import pandas as pd
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import urllib.parse
from psycopg2.extras import execute_values

s = set(stopwords.words('english'))


def load_classifier(classifier):
    """
    Load a classifier from local storage.

    :param classifier: String name of classifier.
    :return: classifier, vectorizer
    """
    fn = os.path.join(os.path.dirname(__file__),
                      'classifiers/{classifier}.clf'.format(classifier=classifier))
    classifier = pickle.load(open(fn, 'rb'))

    fn = os.path.join(os.path.dirname(__file__), 'classifiers/vectorizer.vc')
    vectorizer = pickle.load(open(fn, 'rb'))

    return classifier, vectorizer


def analyze_headlines(headlines):
    """
    Performs semantic analysis on the headline and saves as Headline data-type

    :param headlines: array of Headlines
    :return: array of headlines
    """
    analyzer = SentimentIntensityAnalyzer()

    for headline in headlines:
        vs = analyzer.polarity_scores(headline.headline)

        headline.semantic_value = vs['compound']
        headline.pos = vs['pos']
        headline.neg = vs['neg']
        headline.neu = vs['neu']

    return headlines


def get_headlines(url):
    """
    Gets headlines from http://www.newsapi.org

    :param url: url
    :return: array of headlines
    """
    headlines = []

    req = urllib.request.Request(url)
    response = urllib.request.urlopen(req).read().decode('utf8')
    r = json.loads(response)

    prev_published_at = str(datetime.now()).split(" ")[0]

    for re in r['articles']:

        if str(re['publishedAt']) == 'None':
            published_at = prev_published_at
        else:
            published_at = "" + str(re['publishedAt']).split('T')[0]
            prev_published_at = published_at

        h = Headline(re['title'].split('\n')[0], re['url'], re['source']['id']
                     or re['source']['name'], published_at, re['urlToImage'])
        headlines.append(h)

    return headlines


def save_to_db(headlines):
    """
    Save headlines to database

    :param headlines: all headlines
    :return: null
    """
    db = connect_to_db()
    cur = db.cursor()

    insert_vals = []

    sql = """
        INSERT INTO headlines 
            (headline, predicted_class, link, origin, semantic_value, hashcode, published_at, pos, neg, neu, display_image) 
        VALUES %s
        ON CONFLICT DO NOTHING
        RETURNING id
    """

    for h in headlines:
        insert_vals.append((h.headline, int(h.predicted_class), h.link, h.origin, h.semantic_value, h.sha256(), h.datetime,
                            h.pos, h.neg, h.neu, h.display_image))

    inserted = []
    try:
        inserted = execute_values(cur, sql, insert_vals, fetch=True)
    except Exception as e:
        print("ERROR: {}".format(e))

    print("Stored {count} headline(s)".format(count=len(inserted)))
    db.commit()
    db.close()


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
    clf, v = load_classifier("SVM")

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

    return all_headlines


def main():
    raw_headlines = []
    key = os.environ['NEWS_API_KEY']

    countries = [
        "gb",
        "us"
    ]

    for country in countries:
        lines = get_headlines(
            'https://newsapi.org/v2/top-headlines?country={country}&apiKey={api_key}'.format(api_key=key, country=country))
        raw_headlines.extend(lines)

    all_headlines = analyze_headlines(raw_headlines)
    all_headlines = predict_class(all_headlines)

    if len(sys.argv) > 1:
        if sys.argv[1] == "--persist":
            save_to_db(all_headlines)
        elif sys.argv[1] == "--print":
            print_results(all_headlines)


if __name__ == "__main__":
    main()
