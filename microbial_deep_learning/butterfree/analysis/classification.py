import clusters as c
import tensorflow as tf
import pandas as pd
import re

def one_hot(i, n):
    """ Makes a one-hot vector of length n with 1 in position i. """
    one_hot = [0 for x in range(n)]
    one_hot[i] = 1

    return one_hot

datasets = c.datasets
data_type = 'metaphlan_bugs_list'
body_site = 'stool'

df, dataframes, key_sets = c.__load_data(datasets, data_type, body_site)
cols = [re.sub('_','1',re.sub('\|', '0', x)) for x in df.columns]
df.columns = cols

labels = c.get_labels(dataframes, body_site, key_sets, df)
one_hots = {}
n = len(key_sets)
for i, key in zip(range(n), key_sets):
    one_hots[key] = one_hot(i, n)

lala = [one_hots[label] for label in labels]
lala = pd.Series(lala, index=df.index)
lala = labels.apply(lambda x: "Ascniar" in x).astype(int)
train_fn = tf.estimator.inputs.pandas_input_fn(x=df.iloc[0:4000], y=lala[0:4000], batch_size = 20, shuffle=True)
test_fn = tf.estimator.inputs.pandas_input_fn(x=df.iloc[4000:], y = lala[4000:], batch_size = 20, shuffle=True)

features = [tf.feature_column.numeric_column(x) for x in df.columns[0:-3]]
nn = tf.estimator.DNNClassifier(
    feature_columns=features,
    hidden_units = [100, 100],
    activation_fn = tf.nn.relu,
    n_classes = n,
    dropout = .5,
    )

nn.train(input_fn=train_fn)
nn.evaluate(input_fn=test_fn)
