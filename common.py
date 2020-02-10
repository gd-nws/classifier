import os
import psycopg2
from dotenv import load_dotenv
import time

load_dotenv()


def connect_to_db():
    """
    Creates a connection to a database.

    :return: database
    """

    database = os.environ['DATABASE']
    db_user = os.environ['DB_USER']
    db_password = os.environ['DB_PASSWORD']
    db_host = os.environ['DB_HOST']

    conn = psycopg2.connect(
        "dbname='{dbName}' user='{user}' host='{host}' password='{password}'"
        .format(dbName=database, user=db_user, password=db_password, host=db_host))

    return conn


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
