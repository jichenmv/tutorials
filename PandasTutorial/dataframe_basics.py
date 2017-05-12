from datetime import datetime

import numpy as np
import pandas as pd
import pandas_datareader.data as web

import matplotlib.pyplot as plt


def load_data():

    start = datetime(2010, 1, 1)
    end = datetime(2015, 1, 1)
    return web.DataReader("XOM", "yahoo", start, end)


def fake_data():

    web_stats = {
        'Day': [1, 2, 3, 4, 5, 6],
        'Visitors': [43, 53, 34, 45, 64, 34],
        'Bounce_Rate': [65, 72, 62, 64, 54, 66]
    }
    return pd.DataFrame(web_stats)


if __name__ == '__main__':

    df = load_data()

    print df.shape      # show the dimensions of the data frame
    print df.head()     # show the first 5 rows of the data frame
    print df[['High', 'Low']].head()    # reference specific columns
    print ''

    df2 = fake_data()
    df2.set_index('Day', inplace=True)      # set the specific column as the index
    print df2.shape
    print df2.head()
    print df2.Visitors.tolist()                         # convert the specific column to a list
    print np.array(df2[['Visitors', 'Bounce_Rate']])    # convert the specific columns to an array

    df['Adj Close'].plot()
    plt.show()
