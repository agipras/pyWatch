import streamlit as st

import numpy as np
import pandas as pd
from cryptocmd import CmcScraper
import random
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

cekcek = ['BNB', 'ETH', 'BTC', 'LTC', 'XRP', 'USDT', 'DOGE', 'ABBC', 'BIDR']#, 'UNI']

def cek_hari(cek, hari):
    scraper = CmcScraper(cek)
    df = scraper.get_dataframe()
    close = df["Close"].values[:2000][::-1]
    high = df["High"].values[:2000][::-1]
    low = df["Low"].values[:2000][::-1]
    #close = df["Close"].values[:2000][::-1]

    koleksi_hasil = []
    koleksi_hasil_high = []
    koleksi_hasil_low = []
    for w in range(45, 50):
        for p in range(hari, hari+1):
            x = [(close[_:_+w]-np.min(close[_:_+w]))/(np.max(close[_:_+w])-np.min(close[_:_+w])) for _ in range(close.shape[0]-(w+p-1))]
            y = [(close[_+w+p-1]-np.min(close[_:_+w]))/(np.max(close[_:_+w])-np.min(close[_:_+w])) for _ in range(close.shape[0]-(w+p-1))]

            x_high = [(high[_:_+w]-np.min(high[_:_+w]))/(np.max(high[_:_+w])-np.min(high[_:_+w])) for _ in range(high.shape[0]-(w+p-1))]
            y_high = [(high[_+w+p-1]-np.min(high[_:_+w]))/(np.max(high[_:_+w])-np.min(high[_:_+w])) for _ in range(high.shape[0]-(w+p-1))]

            x_low = [(low[_:_+w]-np.min(low[_:_+w]))/(np.max(low[_:_+w])-np.min(low[_:_+w])) for _ in range(low.shape[0]-(w+p-1))]
            y_low = [(low[_+w+p-1]-np.min(low[_:_+w]))/(np.max(low[_:_+w])-np.min(low[_:_+w])) for _ in range(low.shape[0]-(w+p-1))]

            z = list(zip(x, y))
            random.shuffle(z)
            x, y = zip(*z)
            x = np.asarray(x)
            y = np.asarray(y)

            batas = x.shape[0]*9//10
            x_train = x[:batas]
            y_train = y[:batas]
            x_test = x[batas:]
            y_test = y[batas:]

            z = list(zip(x_high, y_high))
            random.shuffle(z)
            x_high, y_high = zip(*z)
            x_high = np.asarray(x_high)
            y_high = np.asarray(y_high)

            batas = x_high.shape[0]*9//10
            x_train_high = x_high[:batas]
            y_train_high = y_high[:batas]
            x_test_high = x_high[batas:]
            y_test_high = y_high[batas:]

            z = list(zip(x_low, y_low))
            random.shuffle(z)
            x_low, y_low = zip(*z)
            x_low = np.asarray(x_low)
            y_low = np.asarray(y_low)

            batas = x_low.shape[0]*9//10
            x_train_low = x_low[:batas]
            y_train_low = y_low[:batas]
            x_test_low = x_low[batas:]
            y_test_low = y_low[batas:]


            hasil = []
            hasil_high = []
            hasil_low = []
            for k in range(3, 13, 1):
                neigh = KNeighborsRegressor(n_neighbors=k)
                neigh.fit(x_train, y_train)
                hasil.append(mean_absolute_error(neigh.predict(x_test), y_test))

                neigh_high = KNeighborsRegressor(n_neighbors=k)
                neigh_high.fit(x_train_high, y_train_high)
                hasil_high.append(mean_absolute_error(neigh_high.predict(x_test_high), y_test_high))

                neigh_low = KNeighborsRegressor(n_neighbors=k)
                neigh_low.fit(x_train_low, y_train_low)
                hasil_low.append(mean_absolute_error(neigh_low.predict(x_test_low), y_test_low))

            koleksi_hasil.append([w, p, np.argmin(hasil)+3, min(hasil)])
            koleksi_hasil_high.append([w, p, np.argmin(hasil_high)+3, min(hasil)])
            koleksi_hasil_low.append([w, p, np.argmin(hasil_low)+3, min(hasil_low)])
    koleksi_hasil = np.asarray(koleksi_hasil)
    koleksi_hasil_high = np.asarray(koleksi_hasil_high)
    koleksi_hasil_low = np.asarray(koleksi_hasil_low)

    w = int(koleksi_hasil[np.argmin(koleksi_hasil[:,3])][0])
    p = int(koleksi_hasil[np.argmin(koleksi_hasil[:,3])][1])
    w_high = int(koleksi_hasil_high[np.argmin(koleksi_hasil_high[:,3])][0])
    p_high = int(koleksi_hasil_high[np.argmin(koleksi_hasil_high[:,3])][1])
    w_low = int(koleksi_hasil_low[np.argmin(koleksi_hasil_low[:,3])][0])
    p_low = int(koleksi_hasil_low[np.argmin(koleksi_hasil_low[:,3])][1])

    x = [(close[_:_+w]-np.min(close[_:_+w]))/(np.max(close[_:_+w])-np.min(close[_:_+w])) for _ in range(close.shape[0]-(w+p-1))]
    y = [(close[_+w+p-1]-np.min(close[_:_+w]))/(np.max(close[_:_+w])-np.min(close[_:_+w])) for _ in range(close.shape[0]-(w+p-1))]
    x_high = [(high[_:_+w]-np.min(high[_:_+w]))/(np.max(high[_:_+w])-np.min(high[_:_+w])) for _ in range(high.shape[0]-(w+p-1))]
    y_high = [(high[_+w+p-1]-np.min(high[_:_+w]))/(np.max(high[_:_+w])-np.min(high[_:_+w])) for _ in range(high.shape[0]-(w+p-1))]
    x_low = [(low[_:_+w]-np.min(low[_:_+w]))/(np.max(low[_:_+w])-np.min(low[_:_+w])) for _ in range(low.shape[0]-(w+p-1))]
    y_low = [(low[_+w+p-1]-np.min(low[_:_+w]))/(np.max(low[_:_+w])-np.min(low[_:_+w])) for _ in range(low.shape[0]-(w+p-1))]

    z = list(zip(x, y))
    random.shuffle(z)
    x, y = zip(*z)
    x = np.asarray(x)
    y = np.asarray(y)

    z = list(zip(x_high, y_high))
    random.shuffle(z)
    x_high, y_high = zip(*z)
    x_high = np.asarray(x_high)
    y_high = np.asarray(y_high)

    z = list(zip(x_low, y_low))
    random.shuffle(z)
    x_low, y_low = zip(*z)
    x_low = np.asarray(x_low)
    y_low = np.asarray(y_low)

    batas = x.shape[0]*9//10

    x_train = x[:batas]
    y_train = y[:batas]
    x_test = x[batas:]
    y_test = y[batas:]

    x_train_high = x_high[:batas]
    y_train_high = y_high[:batas]
    x_test_high = x_high[batas:]
    y_test_high = y_high[batas:]

    x_train_low = x_low[:batas]
    y_train_low = y_low[:batas]
    x_test_low = x_low[batas:]
    y_test_low = y_low[batas:]

    neigh = KNeighborsRegressor(n_neighbors=int(koleksi_hasil[np.argmin(koleksi_hasil[:,3])][2]))
    neigh.fit(x_train, y_train)
    acc = mean_absolute_error(neigh.predict(x_test), y_test)

    neigh_high = KNeighborsRegressor(n_neighbors=int(koleksi_hasil_high[np.argmin(koleksi_hasil_high[:,3])][2]))
    neigh_high.fit(x_train_high, y_train_high)
    acc = mean_absolute_error(neigh_high.predict(x_test_high), y_test_high)
    
    neigh_low = KNeighborsRegressor(n_neighbors=int(koleksi_hasil_low[np.argmin(koleksi_hasil_low[:,3])][2]))
    neigh_low.fit(x_train_low, y_train_low)
    acc = mean_absolute_error(neigh_low.predict(x_test_low), y_test_low)
    
    return [
      neigh_high.predict(
        ((high[-w:]-np.min(high[-w:]))/(np.max(high[-w:])-np.min(high[-w:]))).reshape(1, -1)
      ) * (np.max(high[-w:])-np.min(high[-w:])) + np.min(high[-w:]),
      neigh.predict(
        ((close[-w:]-np.min(close[-w:]))/(np.max(close[-w:])-np.min(close[-w:]))).reshape(1, -1)
      ) * (np.max(close[-w:])-np.min(close[-w:])) + np.min(close[-w:]),
      neigh_low.predict(
        ((low[-w:]-np.min(low[-w:]))/(np.max(low[-w:])-np.min(low[-w:]))).reshape(1, -1)
      ) * (np.max(low[-w:])-np.min(low[-w:])) + np.min(low[-w:])
    ]
    
def cek_naikTurun(cek):
    st.write(cek)
    my_bar = st.progress(0)
    scraper = CmcScraper(cek)
    df = scraper.get_dataframe()
    close = df["Close"].values[:2000][::-1]

    koleksi_hasil = []
    prog = 1
    for w in range(49, 50):
        for p in range(7, 9):
            #print(w, p)
            my_bar.progress(prog/2)
            prog += 1
            #w = 10
            #p = 7

            x = [(close[_:_+w]-np.min(close[_:_+w]))/(np.max(close[_:_+w])-np.min(close[_:_+w])) for _ in range(close.shape[0]-(w+p-1))]
            y = [(close[_+w+p-1]>close[_+w-1])*1 for _ in range(close.shape[0]-(w+p-1))]

            y_min = np.min(y)

            z = list(zip(x, y))
            random.shuffle(z)
            x, y = zip(*z)
            x = np.asarray(x)
            y = np.asarray(y).astype(int)
            y = y - y_min

            batas = x.shape[0]*9//10
            x_train = x[:batas]
            y_train = y[:batas]
            x_test = x[batas:]
            y_test = y[batas:]

            n_classes = np.max(y)+1

            hasil = []
            for k in range(3, 13, 1):
                neigh = KNeighborsClassifier(n_neighbors=k)
                neigh.fit(x_train, y_train)
                hasil.append(accuracy_score(neigh.predict(x_test), y_test))

            koleksi_hasil.append([w, p, np.argmax(hasil)+3, max(hasil)])
    koleksi_hasil = np.asarray(koleksi_hasil)

    w = int(koleksi_hasil[np.argmax(koleksi_hasil[:,3])][0])
    p = int(koleksi_hasil[np.argmax(koleksi_hasil[:,3])][1])

    x = [(close[_:_+w]-np.min(close[_:_+w]))/(np.max(close[_:_+w])-np.min(close[_:_+w])) for _ in range(close.shape[0]-(w+p-1))]
    y = [(close[_+w+p-1]>close[_+w-1])*1 for _ in range(close.shape[0]-(w+p-1))]

    y_min = np.min(y)

    z = list(zip(x, y))
    random.shuffle(z)
    x, y = zip(*z)
    x = np.asarray(x)
    y = np.asarray(y).astype(int)
    y = y - y_min

    batas = x.shape[0]*9//10
    x_train = x[:batas]
    y_train = y[:batas]
    x_test = x[batas:]
    y_test = y[batas:]

    neigh = KNeighborsClassifier(n_neighbors=int(koleksi_hasil[np.argmax(koleksi_hasil[:,3])][2]))
    neigh.fit(x_train, y_train)
    hasil = confusion_matrix(neigh.predict(x_test), y_test)
    acc = accuracy_score(neigh.predict(x_test), y_test)
    st.write(acc, w, p)
    st.write("stat: " + str(neigh.predict((close[-w:]-np.min(close[-w:]))/(np.max(close[-w:])-np.min(close[-w:])).reshape(1,-1))[0]))
    next_price = cek_hari(cek, 1)
    st.write("1 hari", next_price[0][0], next_price[1][0], next_price[2][0])
    next_price = cek_hari(cek, 7)
    st.write("7 hari", next_price[0][0], next_price[1][0], next_price[2][0])

for cek in cekcek:
    cek_naikTurun(cek)
