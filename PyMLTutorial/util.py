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


def feature_label_extraction(order_items, train_trips):
    """
    Extract features/labels from the training data sets
    :param order_items: (DataFrame)
    :param train_trips: (DataFrame)
    :return: (DataFrame) Data frame with features and labels for training
    """
    trip_df = train_trips.set_index('trip_id')
    trip_df['shopping_hod'] = trip_df.apply(lambda x: compute_hour_of_day(x['shopping_started_at']), axis=1)
    trip_df['shopping_dow'] = trip_df.apply(lambda x: compute_day_of_week(x['shopping_started_at']), axis=1)
    total_item = order_items.groupby('trip_id')['item_id'].count().rename('total_item')
    total_dept = order_items.groupby('trip_id')['department_name'].nunique().rename('total_dept')
    total_quantity = order_items.groupby('trip_id')['quantity'].sum().rename('total_quantity')
    ret = pd.DataFrame(pd.concat((trip_df, total_item, total_dept, total_quantity), axis=1, join='inner'))
    ret['shopping_time'] = ret.apply(
        lambda x: compute_shopping_time(x['shopping_started_at'], x['shopping_ended_at']), axis=1)
    return ret


def feature_extraction(order_items, test_trips):
    """
    Extract features from the testing data sets
    :param order_items: (DataFrame)
    :param test_trips: (DataFrame)
    :return: (DataFrame) Data frame with features for prediction
    """
    trip_df = test_trips.set_index('trip_id')
    trip_df['shopping_hod'] = trip_df.apply(lambda x: compute_hour_of_day(x['shopping_started_at']), axis=1)
    trip_df['shopping_dow'] = trip_df.apply(lambda x: compute_day_of_week(x['shopping_started_at']), axis=1)
    total_item = order_items.groupby('trip_id')['item_id'].count().rename('total_item')
    total_dept = order_items.groupby('trip_id')['department_name'].nunique().rename('total_dept')
    total_quantity = order_items.groupby('trip_id')['quantity'].sum().rename('total_quantity')
    ret = pd.DataFrame(pd.concat((trip_df, total_item, total_dept, total_quantity), axis=1, join='inner'))
    return ret
