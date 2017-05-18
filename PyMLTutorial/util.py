from datetime import datetime
import pandas as pd


def read_data():
    """
    Read data from CSV files 
        1. Order items 
        2. Trips (training set)
        3. Trips (testing set)
    :return: (DataFrame, DataFrame, DataFrame) data frames of the three data sets
    """
    order_items = pd.read_csv('/Users/chen/PycharmProjects/instacart-data/order_items.csv')
    train_trips = pd.read_csv('/Users/chen/PycharmProjects/instacart-data/train_trips.csv')
    test_trips = pd.read_csv('/Users/chen/PycharmProjects/instacart-data/test_trips.csv')
    return order_items, train_trips, test_trips


def compute_hour_of_day(time):
    """
    Compute hour of day from a time string
    :param time: (str)
    :return: (int) hour of day
    """
    ts = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    return ts.hour


def compute_day_of_week(time):
    """
    Compute data of week from a time string
    :param time: (str)
    :return: (int) day of week
    """
    ts = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    return ts.weekday()


def compute_shopping_time(start_time, end_time):
    """
    Compute shopping time in seconds 
    :param start_time: (str) shopping start time
    :param end_time: (str) shopping end time
    :return: (float) total shopping time in seconds
    """
    start = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    return (end - start).total_seconds()
