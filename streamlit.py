import streamlit as st

import numpy as np
import pandas as pd
from cryptocmd import CmcScraper
import random
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

cekcek = ['BNB', 'ETH', 'BTC', 'LTC', 'XRP', 'USDT', 'DOGE', 'ABBC', 'BIDR']#, 'UNI']

for cek in cekcek:
    my_bar = st.progress(0)
    
    st.write(cek + "\n")
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
