import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, metrics
from sklearn.linear_model import LinearRegression
import pickle
import util


def feature_label_extraction(order_items, train_trips):
    """
    Extract features/labels from the training data sets
    :param order_items: (DataFrame)
    :param train_trips: (DataFrame)
    :return: (DataFrame) Data frame with features and labels for training
    """
    trip_df = train_trips.set_index('trip_id')
    trip_df['shopping_hod'] = trip_df.apply(lambda x: util.compute_hour_of_day(x['shopping_started_at']), axis=1)
    trip_df['shopping_dow'] = trip_df.apply(lambda x: util.compute_day_of_week(x['shopping_started_at']), axis=1)
    total_item = order_items.groupby('trip_id')['item_id'].count().rename('total_item')
    total_dept = order_items.groupby('trip_id')['department_name'].nunique().rename('total_dept')
    total_quantity = order_items.groupby('trip_id')['quantity'].sum().rename('total_quantity')
    ret = pd.DataFrame(pd.concat((trip_df, total_item, total_dept, total_quantity), axis=1, join='inner'))
    ret['shopping_time'] = ret.apply(lambda x: util.compute_shopping_time(x['shopping_started_at'], x['shopping_ended_at']), axis=1)
    return ret


def feature_extraction(order_items, test_trips):
    """
    Extract features from the testing data sets
    :param order_items: (DataFrame)
    :param test_trips: (DataFrame)
    :return: (DataFrame) Data frame with features for prediction
    """
    trip_df = test_trips.set_index('trip_id')
    trip_df['shopping_hod'] = trip_df.apply(lambda x: util.compute_hour_of_day(x['shopping_started_at']), axis=1)
    trip_df['shopping_dow'] = trip_df.apply(lambda x: util.compute_day_of_week(x['shopping_started_at']), axis=1)
    total_item = order_items.groupby('trip_id')['item_id'].count().rename('total_item')
    total_dept = order_items.groupby('trip_id')['department_name'].nunique().rename('total_dept')
    total_quantity = order_items.groupby('trip_id')['quantity'].sum().rename('total_quantity')
    ret = pd.DataFrame(pd.concat((trip_df, total_item, total_dept, total_quantity), axis=1, join='inner'))
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
    clf_eval = LinearRegression()
    for trial in range(1, 21, 1):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=trial)
        clf_eval.fit(X_train, y_train)
        y_pred = clf_eval.predict(X_test)
        mae.append(metrics.mean_absolute_error(y_test, y_pred))

    mae = np.array(mae)
    print 'Linear Regression MAEs: '
    print mae.mean(), '+/-', mae.std()

    clf = LinearRegression()
    clf.fit(X, y)
    with open('/Users/chen/PycharmProjects/instacart-data/linear_regression.pickle', 'wb') as f:
        pickle.dump(clf, f)

    '''
    To load the saved model:
    
    pickle_in = open('/Users/chen/PycharmProjects/instacart-data/linear_regression.pickle','rb')
    clf = pickle.load(pickle_in)
    '''

    return clf


def predict_linear_regression(data_frame, feature_list, clf):
    """
    Train simple linear regression models using the features in feature list against the labels in label list
    Evaluate the model accuracy and stability 
    :param data_frame: (DataFrame) containing all testing data
    :param feature_list: (list) feature names
    :param clf: (obj) model 
    :return: (numpy array) the predicted shopping time
    """
    x = np.array(data_frame[feature_list])
    x = preprocessing.scale(x)
    return clf.predict(x)


if __name__ == '__main__':

    order_items_df, train_trips_df, test_trips_df = util.read_data()
    feature_label_df = feature_label_extraction(order_items_df, train_trips_df)
    print 'All training data: '
    print feature_label_df.shape
    print feature_label_df.head()
    print ''

    print 'Linear regression model: '
    features = ['shopping_hod', 'shopping_dow', 'total_item', 'total_dept', 'total_quantity']
    labels = ['shopping_time']
    clf = train_linear_regression(feature_label_df, features, labels)
    print ''

    feature_df = feature_extraction(order_items_df, test_trips_df)
    print 'All prediction data: '
    print feature_df.shape
    print feature_df.head()
    print ''

    print 'Predictions: '
    feature_df['predicted_shopping_time'] = predict_linear_regression(feature_df, features, clf)
    print feature_df[['predicted_shopping_time']].shape
    print feature_df[['predicted_shopping_time']].head()