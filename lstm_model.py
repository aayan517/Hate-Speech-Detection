# lstm_model.py
import re, pickle, nltk, pandas as pd, numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
nltk.download("stopwords")
df=pd.read_csv(r"C:\Users\hello\OneDrive\Desktop\pap ki pari\train.csv")
sw=set(stopwords.words("english"))
def clean(t):
 t=t.lower(); t=re.sub(r"http\S+|@\w+|#","",t); t=re.sub(r"[^a-z ]"," ",t)
 return " ".join([w for w in t.split() if w not in sw])
X=df["tweet"].astype(str).apply(clean); y=df["class"]
tok=Tokenizer(num_words=20000,oov_token="<OOV>"); tok.fit_on_texts(X)
X=pad_sequences(tok.texts_to_sequences(X),maxlen=100)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=Sequential([Embedding(20000,128,input_length=100),LSTM(128,dropout=.2,recurrent_dropout=.2),Dropout(.5),Dense(64,activation="relu"),Dense(3,activation="softmax")])
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
h=model.fit(Xtr,ytr,epochs=10,batch_size=64,validation_split=.1)
pred=np.argmax(model.predict(Xte),1)
print("Accuracy:",accuracy_score(yte,pred))
print(classification_report(yte,pred))
print(confusion_matrix(yte,pred))
model.save("lstm_model.keras")
pickle.dump(tok,open("tokenizer_lstm.pkl","wb"))
plt.plot(h.history["accuracy"]); plt.plot(h.history["val_accuracy"]); plt.legend(["train","val"]); plt.show()
