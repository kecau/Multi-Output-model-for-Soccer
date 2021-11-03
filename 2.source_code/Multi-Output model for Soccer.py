import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
import matplotlib.pyplot as plt

dataframe = pd.read_excel('Data.xlsx',  sheet_name = ['20-21-test','19-20', '18-19', '17-18', '16-17', '15-16', '14-15', '13-14', '12-13', '11-12', '10-11'])
dataframe = pd.concat(dataframe, axis= 0, ignore_index=True)

CSV_HEADER = ["FW0", "FW0B", "FW1" ,'AMF0', 'AMF0B', 'AMF1', 'AMF1B',
       'Wing0', 'Wing0B', 'Wing1', 'Wing1B', 'CMF0', 'CMF0B', 'CMF0C', 'CMF1',
       'CMF1B', 'DMF0', 'DMF1', 'DMF1B', 'WB0', 'WB0B', 'WB1', 'WB1B', 'CB0',
       'CB0B', 'CB1', 'CB1B', 'GK0', 'GK1', 'Opp_Level', 'Season','Ball_pos','match_oder', 'oppense_team', 'Formation', 'Win', 'Style','sum']


CATEGORICAL_DATA = ['match_oder', 'oppense_team','Opp_Level', 'Season' ]


CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    feature_name: sorted([str(value) for value in list(dataframe[feature_name].unique())])
    for feature_name in CSV_HEADER
    if feature_name
    in list(CATEGORICAL_DATA)
}

NUMERIC_FEATURE_NAMES = ['Ball_pos']
BINARY_FEATURE_NAMES = ["FW0", "FW0B", "FW1" ,'AMF0', 'AMF0B', 'AMF1', 'AMF1B',
       'Wing0', 'Wing0B', 'Wing1', 'Wing1B', 'CMF0', 'CMF0B', 'CMF0C', 'CMF1',
       'CMF1B', 'DMF0', 'DMF1', 'DMF1B', 'WB0', 'WB0B', 'WB1', 'WB1B', 'CB0',
       'CB0B', 'CB1', 'CB1B', 'GK0', 'GK1']



FEATURE_NAMES = NUMERIC_FEATURE_NAMES + BINARY_FEATURE_NAMES + list(
    CATEGORICAL_FEATURES_WITH_VOCABULARY.keys()
)

dataframe_formation = dataframe['Formation'].unique().tolist()
dataframe_formation_to_encoded = {x: i for i, x in enumerate(dataframe_formation)}
dataframe_index_to_formation = {i: x for i, x in enumerate(dataframe_formation)}
dataframe['Formation'] = dataframe['Formation'].map(dataframe_formation_to_encoded)



win_cate = dataframe['Win'].unique().tolist()
win_to_encoded = {x: i for i, x in enumerate(win_cate)}
index_to_win = {i: x for i, x in enumerate(win_cate)} 
dataframe['Win'] = dataframe['Win'].map(win_to_encoded)


train_data = train_dt[["FW0", "FW0B", "FW1" ,'AMF0', 'AMF0B', 'AMF1', 'AMF1B',
       'Wing0', 'Wing0B', 'Wing1', 'Wing1B', 'CMF0', 'CMF0B', 'CMF0C', 'CMF1',
       'CMF1B', 'DMF0', 'DMF1', 'DMF1B', 'WB0', 'WB0B', 'WB1', 'WB1B', 'CB0',
       'CB0B', 'CB1', 'CB1B', 'GK0', 'GK1', 'Ball_pos']].values.astype(np.float)

train_data_cate = train_dt[['Opp_Level', 'Season','match_oder', 'oppense_team']].values


test_data = test_dt[["FW0", "FW0B", "FW1" ,'AMF0', 'AMF0B', 'AMF1', 'AMF1B',
       'Wing0', 'Wing0B', 'Wing1', 'Wing1B', 'CMF0', 'CMF0B', 'CMF0C', 'CMF1',
       'CMF1B', 'DMF0', 'DMF1', 'DMF1B', 'WB0', 'WB0B', 'WB1', 'WB1B', 'CB0',
       'CB0B', 'CB1', 'CB1B', 'GK0', 'GK1', 'Ball_pos']].values.astype(np.float)

test_data_cate = test_dt[['Opp_Level', 'Season','match_oder', 'oppense_team']].values


y_formation = train_dt["Formation"].values
y_style = train_dt['Style'].values
y_win = train_dt['Win'].values


y_test_formation = test_dt["Formation"].values
y_test_style = test_dt['Style'].values
y_test_win = test_dt['Win'].values

def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES + BINARY_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.float32
            )
        else:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.string
            )
    return inputs


def encode_inputs(inputs, encoding_size):
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a lookup to convert a string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            index = StringLookup(
                vocabulary=vocabulary, mask_token=None, num_oov_indices=0
            )
            # Convert the string input values into integer indices.
            value_index = index(inputs[feature_name])
            # Create an embedding layer with the specified dimensions
            embedding_ecoder = layers.Embedding(
                input_dim=len(vocabulary), output_dim=encoding_size
            )
            # Convert the index values to embedding representations.
            encoded_feature = embedding_ecoder(value_index)
        elif feature_name in NUMERIC_FEATURE_NAMES:
            encoded_feature = tf.expand_dims(inputs[feature_name], -1)
            encoded_feature = layers.Dense(units=encoding_size)(encoded_feature)
            encoded_feature = layers.Dense(units=encoding_size)(encoded_feature)
            encoded_feature = layers.Dense(units=encoding_size)(encoded_feature)

        else:
            # Project the numeric feature to encoding_size using linear transformation.
            encoded_feature = tf.expand_dims(inputs[feature_name], -1)
            encoded_feature = layers.Dense(units=encoding_size)(encoded_feature)
        encoded_features.append(encoded_feature)
    return encoded_features


def create_model(encoding_size):
    inputs = create_model_inputs()
    feature_list = encode_inputs(inputs, encoding_size)
    concat = layers.concatenate(feature_list)

    x_1_input = layers.Dense(42, kernel_regularizer=keras.regularizers.l2(0.001))(concat)
    x_1 = layers.BatchNormalization()(x_1_input)
    x_1 = layers.Dropout(dropout_rate)(x_1)
    x_1 = layers.ReLU()(x_1)
    x_1 = layers.Dense(42, kernel_regularizer=keras.regularizers.l2(0.001))(x_1) + x_1_input
    output_for = layers.Dense(31, activation='softmax', name='output_for')(x_1)

    x_2_input = layers.Dense(42, kernel_regularizer=keras.regularizers.l2(0.001))(concat)
    x_2 = layers.BatchNormalization()(x_2_input)
    x_2 = layers.Dropout(dropout_rate)(x_2)
    x_2 = layers.ReLU()(x_2)
    x_2 = layers.Dense(42, kernel_regularizer=keras.regularizers.l2(0.001))(x_2) + x_2_input
    output_style = layers.Dense(2, activation='softmax', name='output_style')(x_2)

    x_3_input = layers.Dense(42, kernel_regularizer=keras.regularizers.l2(0.001))(concat)
    x_3 = layers.BatchNormalization()(x_3_input)
    x_3 = layers.Dropout(dropout_rate)(x_3)
    x_3 = layers.ReLU()(x_3)
    x_3 = layers.Dense(42, kernel_regularizer=keras.regularizers.l2(0.001))(x_3) + x_3_input
    x_3 = layers.Dense(24, kernel_regularizer=keras.regularizers.l2(0.001))(x_3)
    x_3 = layers.BatchNormalization()(x_3)
    x_3 = layers.Dropout(dropout_rate)(x_3)
    x_3 = layers.ReLU()(x_3)
    x_3 = layers.Dense(12)(x_3)
    x_3 = layers.BatchNormalization()(x_3)
    x_3 = layers.Dropout(dropout_rate)(x_3)
    x_3 = layers.ReLU()(x_3)
    output_win = layers.Dense(3, activation='softmax', name='output_win')(x_3)

    model = keras.Model(inputs=inputs, outputs = [output_for, output_style, output_win])
    return model


model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
              loss= {'output_for': keras.losses.SparseCategoricalCrossentropy(), 
                      'output_win':keras.losses.SparseCategoricalCrossentropy(), 
                      'output_style':keras.losses.SparseCategoricalCrossentropy()},
              metrics =  {'output_for': [keras.metrics.SparseCategoricalAccuracy()],
                      'output_win':[keras.metrics.SparseCategoricalAccuracy()], 
                      'output_style':[keras.metrics.SparseCategoricalAccuracy()] }
              
              # {'output_for': [keras.losses.SparseCategoricalCrossentropy() ,keras.metrics.Precision(name='precision'),keras.metrics.Recall(name='recall')],
              #         'output_win':[keras.losses.SparseCategoricalCrossentropy(),keras.metrics.Precision(name='precision'),keras.metrics.Recall(name='recall')], 
              #         'output_style':[keras.losses.SparseCategoricalCrossentropy() ,keras.metrics.Precision(name='precision'),keras.metrics.Recall(name='recall')]}
)


