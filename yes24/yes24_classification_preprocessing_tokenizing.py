import nltk
import pickle
import numpy as np
import pandas as pd
from ko_restoration import main
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import *
from tensorflow.keras.preprocessing.sequence import pad_sequences

def visualize_process(current, total):
    if (current % 250 == 0) and (current>1):
        print('.', end='')
    if current % 5000 == 0:
        print('{} / {}'.format(current, total))

raw_df = pd.read_csv('/Users/san/work/python/Deep_Learning/LSTM_DNN_PJT/data/raw_data.csv', index_col=0)
print('raw_df : \n', raw_df.head(5))

print('\n{:=^30}'.format('gap handling'))
# for i in range(len(raw_df)): # 공백 하나만 남기기
#     raw_df.iloc[i,3] = ' '.join(raw_df.iloc[i,3].split())
#     visualize_process(i, len(raw_df))
raw_df['Introduction'] = raw_df['Introduction'].apply(lambda x : ' '.join(x.split()))
print('after gap-healing:', len(raw_df))

# 중복된 data 제거(row)
df = raw_df.drop_duplicates(subset=['Introduction'])
print('after dropna', len(df))

# 새로운 index 할당
df.reset_index(drop=True, inplace=True) # drop=True : 기존 index를 제거

# ## Book Introduction Preprocessing
# ## data를 X, Y로 분할
X = df['Introduction'].copy()
Y = df['Medium_category']

# ## Y(label) 처리
encoder = LabelEncoder()
labeled_Y = encoder.fit_transform(Y)    # Y값 label Encoding
label = encoder.classes_

# encoding mapping 정보를 저장
with open('/Users/san/work/python/Deep_Learning/LSTM_DNN_PJT/data/category_encoder_12.pickle', 'wb') as f:
  pickle.dump(encoder, f)

# label을 onehot encoding으로 변환
onehot_Y = to_categorical(labeled_Y)

# ## X(data) 처리
# ### 형태소 분석
print('\n{:=^30}'.format('형태소 분석'))
tokenizer = 'komoran'
komoran, complex_verb_set = main.set_env(tokenizer)
for i in range(len(X)):
    X[i] = main.start_restoration(komoran, complex_verb_set, [X[i]]).split()
    visualize_process(i, len(X))
print('\n{}_restoration_len_500 :\n'.format(tokenizer), X[:5])
print('X.shape :', X.shape)
print('len(X) :', len(X))

print('\n{:=^30}'.format('cut over 500 token'))
limit = 100
for i in range(len(X)):
    if len(X[i]) >=limit:
        X[i] = X[i][:limit]
    visualize_process(i, len(X))

# 형태소 분석 된 X data 저장
X.to_csv('../data/{}_restoration_len_{}.csv'.format((tokenizer, limit)))

# 불용어 제거
kor_stopwords = pd.read_csv('../data/stopwords.csv')
nltk.download('stopwords')
eng_stopwords = set(stopwords.words('english'))
stopword = list(kor_stopwords['stopword']) + list(eng_stopwords)
# 불용어 제거 후 형태소로 이루어진 문장으로 재조합
print('\n{:=^30}'.format('delete one letter token'))
for i in range(len(X)) :
    result = []
    for j in range(len(X[i])):
        if len(X[i][j]) > 1:  # 길이가 한 글자인 것은 지움
            if X[i][j] not in stopword:
                result.append(X[i][j])
    X[i] = ' '.join(result)
    visualize_process(i, len(X))
print('\n{}_restoration_stopwords_len_{} :\n'.format(tokenizer, limit), X[:5])


# In[ ]:


# 불용어 제거된 X data 저장
X.to_csv('../data/{}_restoration_stopwords_len_{}.csv'.format(tokenizer, limit))


# ### 토크나이징
# tokenizing : 각 형태소에 숫자 label값을 배정
token = Tokenizer()
token.fit_on_texts(X)  # 형태소에 어떤 숫자를 배정할 것인지
tokened_X = token.texts_to_sequences(X)  # 토큰에 저장된 label을 바탕으로 문장(X)을 변환
print('tokened_X :\n', tokened_X[:5])
# token 저장
# 데이터 형태 그대로 저장
with open('../data/cat_12_{}_len_{}_book_token.pickle'.format(tokenizer, limit), 'wb') as f:
  pickle.dump(token, f)


# ## data 확인
# 형태소 개수 확인
wordsize = len(token.word_index) + 1
# print('word index : ', token.word_index)
print('wordsize is : ', wordsize)  # index 0를 padding 으로 추가 예정

max = 0
for i in range(len(tokened_X)):
  if max < len(tokened_X[i]):
      max = len(tokened_X[i])
print('max token len : ', max)

# padding
X_pad = pad_sequences(tokened_X, max) # 앞쪽을 0으로 채움
print('X_pad :\n', X_pad[:5])

# ## Train, Test set split
X_train, X_test, Y_train, Y_test = train_test_split(X_pad, onehot_Y, test_size=0.2)
print('X_train.shape :', X_train.shape)
print('X_test.shape :', X_test.shape)
print('Y_train.shape :', Y_train.shape)
print('Y_test.shape :', Y_test.shape)

# ## Train, Test set 저장
xy = X_train, X_test, Y_train, Y_test
np.save('../data/book_data_tok_{}_max_{}_wordsize_{}'.format(tokenizer, max, wordsize), xy)