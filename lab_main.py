# загрузим библиотеки и установим опции
from __future__ import division, print_function
# отключим всякие предупреждения Anaconda
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# загрузим обучающую и тестовую выборки
train_df = pd.read_csv('data/train_sessions.csv')#,index_col='session_id')
test_df = pd.read_csv('data/test_sessions.csv')#, index_col='session_id')

# приведем колонки time1, ..., time10 к временному формату
times = ['time%s' % i for i in range(1, 11)]
train_df[times] = train_df[times].apply(pd.to_datetime)
test_df[times] = test_df[times].apply(pd.to_datetime)

# отсортируем данные по времени
train_df = train_df.sort_values(by='time1')

# посмотрим на заголовок обучающей выборки
train_df.head()

sites = ['site%s' % i for i in range(1, 11)]
#заменим nan на 0
train_df[sites] = train_df[sites].fillna(0).astype('int').astype('str')
test_df[sites] = test_df[sites].fillna(0).astype('int').astype('str')


#создадим тексты необходимые для обучения word2vec
train_df['list'] = train_df['site1']
test_df['list'] = test_df['site1']
for s in sites[1:]:
    train_df['list'] = train_df['list']+","+train_df[s]
    test_df['list'] = test_df['list']+","+test_df[s]
train_df['list_w'] = train_df['list'].apply(lambda x: x.split(','))
test_df['list_w'] = test_df['list'].apply(lambda x: x.split(','))


from gensim.models import word2vec

#объединим обучающую и тестовую выборки и обучим нашу модель на всех данных
#с размером окна в 6=3*2 (длина предложения 10 слов) и итоговыми векторами размерности 300, параметр workers отвечает за количество ядер
test_df['target'] = -1
data = pd.concat([train_df, test_df], axis=0)
model = word2vec.Word2Vec(data['list_w'], vector_size=300, window=3, workers=4)


#создадим словарь со словами и соответсвующими им векторами
w2v = dict(zip(model.wv.index_to_key, model.wv))

class mean_vectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(next(iter(w2v.values())))

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

data_mean=mean_vectorizer(w2v).fit(train_df['list_w']).transform(train_df['list_w'])

# подключим библиотеки keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


def split(train,y,ratio):
    idx = round(train.shape[0] * ratio)
    return train[:idx, :], train[idx:, :], y[:idx], y[idx:]
y = train_df['target']
Xtr, Xval, ytr, yval = split(data_mean, y, 0.8)
Xtr.shape, Xval.shape, ytr.mean(), yval.mean()



# опишем нейронную сеть
model = Sequential()
model.add(Dense(128, input_dim=(Xtr.shape[1])))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])

history = model.fit(Xtr, ytr,
                    batch_size=128,
                    epochs=15,
                    )


classes = model.predict(Xval, batch_size=128)

print(roc_auc_score(yval, classes))