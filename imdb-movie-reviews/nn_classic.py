import re
import pandas as pd
from sklearn.model_selection import train_test_split
 
def clean_review(text):
    # Strip HTML tags
    text = re.sub('<[^<]+?>', ' ', text)
 
    # Strip escaped quotes
    text = text.replace('\\"', '')
 
    # Strip quotes
    text = text.replace('"', '')


 
    return text
 
df = pd.read_csv('./Dataset/train/train.csv', names=['review','label'])
# print(df.head()
df['cleaned_review'] = df['review'].apply(clean_review)
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['label'], test_size=0.2)



from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
 
vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'), 
                             lowercase=True, min_df=3, max_df=0.9, max_features=5000)
X_train_onehot = vectorizer.fit_transform(X_train)



from keras.models import Sequential
from keras.layers import Dense
 

print(vectorizer.get_feature_names())

model = Sequential()
 
model.add(Dense(units=500, activation='relu', input_dim=len(vectorizer.get_feature_names())))
model.add(Dense(units=250, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


model.fit(X_train_onehot[:-100], y_train[:-100], 
          epochs=2, batch_size=128, verbose=1, 
          validation_data=(X_train_onehot[-100:], y_train[-100:]))


scores = model.evaluate(vectorizer.transform(X_test), y_test, verbose=1)
print("Accuracy:", scores[1]) 