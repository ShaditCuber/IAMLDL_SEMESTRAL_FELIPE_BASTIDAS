import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, SimpleRNN
from keras.models import Model
import numpy as np
from keras.layers import Bidirectional, GlobalMaxPool1D
from transformers import DistilBertTokenizer, TFDistilBertModel
import logging
from joblib import dump
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense



logger = logging.getLogger("iamldl_logs")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler("iamldl_logs.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def show_confusion_matrix(confusion_matrix, model_name):
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[1, 2, 3, 4, 5, 6, 7],
        yticklabels=[1, 2, 3, 4, 5, 6, 7],
    )
    plt.xlabel("Predicciones")
    plt.ylabel("Valores Reales")
    plt.title("Matriz de Confusión {}".format(model_name))
    plt.savefig("matrices_de_confusion/{}.png".format(model_name))


os.makedirs("models", exist_ok=True)
os.makedirs("matrices_de_confusion", exist_ok=True)

logger.info("Cargando el dataset desde un archivo JSON")
file_path = "dataset.json"
df = pd.read_json(file_path)
df.head()

logger.info("Cantidad de datos: {}".format(df.shape[0]))

"""
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('spanish')

stop_words = [
    "a",
    "actualmente",
    "acuerdo",
    "adelante",
    "ademas",
    "adrede",
    "afirmó",
    "agregó",
    "ahi",
    "ahora",
    "ahí",
    "al",
    "algo",
    "alguna",
    "algunas",
    "alguno",
    "algunos",
    "algún",
    "alli",
    "alrededor",
    "ambos",
    "ampleamos",
    "antano",
    "antaño",
    "ante",
    "anterior",
    "antes",
    "apenas",
    "aproximadamente",
    "aquel",
    "aquella",

]
"""

def clean_text(text)->str:
    """Función para limpiar los textos de los Informes.

    :param text: Informe a limpiar
    :type text: str
    :return: Informe limpio de caracteres especiales, saltos de línea y caracteres de puntuación u otros caracteres no alfanuméricos.
    :rtype: str
    """
    replaces = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}
    for k, v in replaces.items():
        text = text.replace(k, v)

    text = text.replace("\n", " ").replace("\t", " ")
    text = text.replace('"', "'")
    text = text.replace("'", " ")
    text = text.replace("“", " ")
    text = text.replace("”", " ")
    text = text.replace("-", "")
    text = text.replace("●", "")
    text = text.replace("•", "")
    text = text.replace("►", "")
    text = text.replace("▲", "")
    text = text.replace(".", "")
    text = text.replace(":", "")

    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text.lower()




df["Texto"] = df["Texto"].apply(clean_text)



X_train, X_test, y_train, y_test = train_test_split(df["Texto"], df["Nota"], test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)


logger.info("Árbol de Decisión")
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_vect, y_train)
y_pred_dt = model_dt.predict(X_test_vect)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
logger.info("Accuracy: %.2f%%" % (accuracy_dt * 100.0))
cm_dt = confusion_matrix(y_test, y_pred_dt, labels=[1, 2, 3, 4, 5, 6, 7])
show_confusion_matrix(cm_dt, "Árbol-de-Decisión")
dump(model_dt, "models/dt.joblib")


logger.info("SVM")
model_svm = SVC()
model_svm.fit(X_train_vect, y_train)
y_pred_svm = model_svm.predict(X_test_vect)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
logger.info("Accuracy: %.2f%%" % (accuracy_svm * 100.0))
cm_svm = confusion_matrix(y_test, y_pred_svm, labels=[1, 2, 3, 4, 5, 6, 7])
show_confusion_matrix(cm_svm, "SVM")
dump(model_svm, "models/svm.joblib")

logger.info("Naive Bayes")
model_nb = MultinomialNB()
model_nb.fit(X_train_vect, y_train)
y_pred_nb = model_nb.predict(X_test_vect)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
logger.info("Accuracy: %.2f%%" % (accuracy_nb * 100.0))
cm_nb = confusion_matrix(y_test, y_pred_nb, labels=[1, 2, 3, 4, 5, 6, 7])
show_confusion_matrix(cm_nb, "MultinomialNB")
dump(model_nb, "models/nb.joblib")




logger.info("Redes Neuronales Recurrentes")
logger.info("Preprocesamiento de los textos con Tokenizer")
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
logger.info("Conversión de los textos a secuencias")
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
logger.info("Padding de las secuencias")
X_train_pad = pad_sequences(X_train_seq, maxlen=100, padding="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=100, padding="post")

max_features = 5000
maxlen = 100
inp = Input(shape=(maxlen,))
embed_size = 64
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(LSTM(50, return_sequences=True, name="lstm_layer"))(x)
x = SimpleRNN(30, return_sequences=True, name="rnn_layer")(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.3)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(7, activation="softmax")(x)

model = Model(inputs=inp, outputs=x)

model.compile(
    loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"]
)

y_train_adj = y_train - 1
y_test_adj = y_test - 1


model.fit(
    X_train_pad,
    y_train_adj,
    batch_size=32,
    epochs=20,
    validation_data=(X_test_pad, y_test_adj),
)

y_pred_rnn = model.predict(X_test_pad)
y_pred_rnn = [np.argmax(pred) + 1 for pred in y_pred_rnn]
accuracy_rnn = accuracy_score(y_test, y_pred_rnn)
logger.info("Accuracy: %.2f%%" % (accuracy_rnn * 100.0))
cm_rnn = confusion_matrix(y_test, y_pred_rnn, labels=[1, 2, 3, 4, 5, 6, 7])
show_confusion_matrix(cm_rnn, "Redes-Neuronales-Recurrentes")
model.save("models/rnn.keras")



logger.info("BERT")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
distilbert = TFDistilBertModel.from_pretrained("distilbert-base-multilingual-cased")

def tokenizar(textos, max_len=128):
    input_ids = []
    attention_masks = []

    for texto in textos:
        encoded = tokenizer.encode_plus(
            texto,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="tf",
            truncation=True,
        )

        input_ids.append(encoded["input_ids"][0])
        attention_masks.append(encoded["attention_mask"][0])

    return np.array(input_ids), np.array(attention_masks)


X_train_ids, X_train_masks = tokenizar(X_train, max_len=128)
X_test_ids, X_test_masks = tokenizar(X_test, max_len=128)


logger.info("Construcción del modelo")
input_ids = Input(shape=(128,), dtype="int32", name="input_ids")
attention_masks = Input(shape=(128,), dtype="int32", name="attention_masks")

logger.info("Capa de DistilBERT")
distilbert_output = distilbert([input_ids, attention_masks])[0]
cls_token = distilbert_output[:, 0, :]

logger.info("Capas de la red neuronal")
x = Dropout(0.3)(cls_token)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(7, activation="softmax")(x)
model = Model(inputs=[input_ids, attention_masks], outputs=output)

for layer in model.layers[:20]:
    layer.trainable = False

for layer in model.layers[20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00002), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

y_train_adj = y_train - 1
y_test_adj = y_test - 1

history = model.fit(
    [X_train_ids, X_train_masks],
    y_train_adj,
    batch_size=16,
    epochs=10,
    validation_data=([X_test_ids, X_test_masks], y_test_adj),
)

y_pred_bert = model.predict([X_test_ids, X_test_masks])
y_pred_bert = [np.argmax(pred) + 1 for pred in y_pred_bert]
accuracy_bert = accuracy_score(y_test, y_pred_bert)
logger.info("Accuracy: %.2f%%" % (accuracy_bert * 100.0))
cm_bert = confusion_matrix(y_test, y_pred_bert, labels=[1, 2, 3, 4, 5, 6, 7])
show_confusion_matrix(cm_bert, "BERT")
model.save("models/bert.keras")
