import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from feature_extraction import feature_df
from sklearn.ensemble import RandomForestClassifier
SEED = 21
feature_df["edge"] = feature_df["edge"].apply(lambda x: x.split("_")[0])
feature_df["key_cosine_similarity"].fillna(0, inplace=True)
feature_df["venue_cosine_similarity"].fillna(0, inplace=True)
train_dataset = feature_df
test_dataset = feature_df.drop(feature_df.sample(frac=0.8, random_state=SEED).index)
# sns.pairplot(train_dataset, diag_kind="kde")
# plt.show()

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('edge').apply(pd.to_numeric)
test_labels = test_dataset.pop('edge').apply(pd.to_numeric)


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# def build_model():
#     model = keras.Sequential([
#         layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
#         layers.Dense(64, activation='relu'),
#         layers.Flatten(),
#         layers.Dense(1, activation='softmax')
#     ])
#
#     optimizer = tf.keras.optimizers.RMSprop(0.01)
#
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=optimizer,
#                   metrics=['mae', 'mse', "accuracy"])
#     return model
#
#
# model = build_model()
# EPOCHS = 1000
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# history = model.fit(
#     normed_train_data, train_labels,
#     epochs=EPOCHS, validation_split=0.2, verbose=1,
#     callbacks=[early_stop])
# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# hist.tail()
model = RandomForestClassifier(max_features="sqrt")
model.fit(normed_train_data, train_labels)
preds = model.predict(normed_test_data)
print(roc_auc_score(test_labels, preds[:, 1]))
# estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
# kfold = KFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
