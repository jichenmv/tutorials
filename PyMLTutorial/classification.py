import numpy as np
from sklearn import model_selection, metrics
from sklearn import neighbors
import pickle
import util


def train_knn(data_frame, feature_list, label_list):
    """
    Train k nearest neighbor classification models using the features in feature list against the labels in label list
    Evaluate the model accuracy and stability 
    :param data_frame: (DataFrame) containing all training data
    :param feature_list: (list) feature names
    :param label_list: (list) label names
    :return: 
    """
    X = np.array(data_frame[feature_list])
    y = np.array(data_frame[label_list])

    score = []
    clf_eval = neighbors.KNeighborsClassifier()
    for trial in range(1, 21, 1):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=trial)
        clf_eval.fit(X_train, y_train.ravel())
        y_pred = clf_eval.predict(X_test)
        score.append(metrics.accuracy_score(y_test, y_pred))

    score = np.array(score)
    print 'K Nearest Neighbor accuracy score: '
    print score.mean(), '+/-', score.std()

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X, y.ravel())
    with open('/Users/chen/PycharmProjects/instacart-data/knn_classification.pickle', 'wb') as f:
        pickle.dump(clf, f)

    '''
    To load the saved model:

    pickle_in = open('/Users/chen/PycharmProjects/instacart-data/knn_classification.pickle','rb')
    clf = pickle.load(pickle_in)
    '''

    return clf


def predict_knn(data_frame, feature_list, clf):
    """
    Use the KNN model to make a prediction
    :param data_frame: (DataFrame) containing all testing data
    :param feature_list: (list) feature names
    :param clf: (obj) model 
    :return: (numpy array) the predicted fulfillment_model
    """
    x = np.array(data_frame[feature_list])
    return clf.predict(x)


if __name__ == '__main__':
    order_items_df, train_trips_df, test_trips_df = util.read_data()
    feature_label_df = util.feature_label_extraction(order_items_df, train_trips_df)
    print 'All training data: '
    print feature_label_df.shape
    print feature_label_df.head()
    print ''

    print 'KNN classification model: '
    features = ['shopper_id', 'store_id']
    labels = ['fulfillment_model']
    clf = train_knn(feature_label_df, features, labels)
    print ''

    feature_df = util.feature_extraction(order_items_df, test_trips_df)
    print 'All prediction data: '
    print feature_df.shape
    print feature_df.head()
    print ''

    print 'Predictions: '
    feature_df['predicted_fulfillment_model'] = predict_knn(feature_df, features, clf)
    print feature_df[['predicted_fulfillment_model']].shape
    print feature_df[['predicted_fulfillment_model']].head()
