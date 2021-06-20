import os
from dotenv import load_dotenv
import time
import pickle
from nltk.corpus import stopwords
from pymongo import MongoClient
import string


load_dotenv()
s = set(stopwords.words('english'))

def connect_to_mongo_db():
  """
  Connect to a mongo database.

  :return: Mongo database connection.
  """
  client = MongoClient(os.environ['CONNECTION_STRING'])

  # db=client.admin
  # # Issue the serverStatus command and print the results
  # serverStatusResult=db.command("serverStatus")
  # print(serverStatusResult)

  db = client.GoodNews
  return db, client


def normalise_column(df, column):
    """
    Normalise a column of floats

    :param df: data frame
    :param column: column name
    :return: data frame with update column
    """

    max_c = df[column].max()
    min_c = df[column].min()

    if max_c == min_c:
        df[column] = df[column].apply(lambda x: 0)
    else:
        df[column] = df[column].apply(lambda x: float(
            (x - min_c)) / float((max_c - min_c)))

    return df


def to_epoch(date_time):
    """
    Convert datetime to epoch

    :param date_time: datetime
    :return: epoch
    """
    pattern = '%Y-%m-%d'
    epoch = int(time.mktime(time.strptime(str(date_time), pattern)))
    return epoch


def load_file(path):
    """
    Load a file

    :param path: File path
    :return: file
    """
    fn = os.path.join(os.path.dirname(__file__), path)
    return pickle.load(open(fn, 'rb'))


def filter_stop_words(text):
    """
    Filter all stop words from a string to reduce headline size.

    :param text: text to filter
    :return: shortened headline
    """
    words = filter(lambda w: not w in s, text.split())
    line = ""
    l = 0
    for w in words:
        if l < 20:
            line += w + " "
            l += 1
        else:
            return line.strip()
    return line.strip()


def strip_punctuation(text):
    """
    Remove all punctuation from text

    :param text: raw headline
    :return: parsed headline
    """
    if text is None:
        text = ""

    return text.translate(str.maketrans('', '', string.punctuation))
