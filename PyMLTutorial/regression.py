from datetime import datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, metrics
from sklearn.linear_model import LinearRegression


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


def feature_extraction(order_items, train_trips):
    """
    Extract features/labels from the data sets
    :param order_items: (DataFrame)
    :param train_trips: (DataFrame)
    :return: (DataFrame) Data frame with features and labels
    """
    trip_df = train_trips.set_index('trip_id')
    trip_df['shopping_hod'] = trip_df.apply(lambda x: compute_hour_of_day(x['shopping_started_at']), axis=1)
    trip_df['shopping_dow'] = trip_df.apply(lambda x: compute_day_of_week(x['shopping_started_at']), axis=1)
    total_item = order_items.groupby('trip_id')['item_id'].count().rename('total_item')
    total_dept = order_items.groupby('trip_id')['department_name'].nunique().rename('total_dept')
    total_quantity = order_items.groupby('trip_id')['quantity'].sum().rename('total_quantity')
    ret = pd.DataFrame(pd.concat((trip_df, total_item, total_dept, total_quantity), axis=1, join='inner'))
    ret['shopping_time'] = ret.apply(lambda x: compute_shopping_time(x['shopping_started_at'], x['shopping_ended_at']), axis=1)
    return ret


def train_linear_regression(data_frame, feature_list, label_list):
    """
    Train simple linear regression models using the features in feature list against the labels in label list
    Evaluate the model accuracy and stability 
    :param data_frame: (DataFrame) containing all training data
    :param feature_list: (list) feature names
    :param label_list: (list) label names
    :return: 
    """
    X = np.array(data_frame[feature_list])
    X = preprocessing.scale(X)
    y = np.array(data_frame[label_list])

    mae = []
    clf = LinearRegression()
    for trial in range(1, 21, 1):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=trial)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        mae.append(metrics.mean_absolute_error(y_test, y_pred))

    mae = np.array(mae)
    print 'Linear Regression MAEs: '
    print mae.mean(), '+/-', mae.std()


if __name__ == '__main__':

    order_items_df, train_trips_df, test_trips_df = read_data()
    feature_label_df = feature_extraction(order_items_df, train_trips_df)
    print 'All training data: '
    print feature_label_df.shape
    print feature_label_df.head()
    print ''

    features = ['shopping_hod', 'shopping_dow', 'total_item', 'total_dept', 'total_quantity']
    labels = ['shopping_time']
    train_linear_regression(feature_label_df, features, labels)
    print ''
