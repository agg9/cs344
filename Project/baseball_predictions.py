from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

# Read in the CSV files
batting = pd.read_csv('Batting.csv')
people = pd.read_csv('People.csv')

df = pd.merge(batting, people[['playerID', 'nameFirst', 'nameLast']], on='playerID', how="left")

# Create Batting Average (`AVE`) column
df['AVE'] = df['H'] / df['AB']

# Create On Base Percent (`OBP`) column
plate_appearances = (df['AB'] + df['BB'] + df['SF'] + df['SH'] + df['HBP'])
df['OBP'] = (df['H'] + df['BB'] + df['HBP']) / plate_appearances

# Create Slugging Percent (`Slug_Percent`) column
df['single'] = ((df['H'] - df['2B']) - df['3B']) - df['HR']
single = ((df['H'] - df['2B']) - df['3B']) - df['HR']
df['Slug_Percent'] = ((df['HR'] * 4) + (df['3B'] * 3) + (df['2B'] * 2) + single) / df['AB']

# Create On Base plus Slugging Percent (`OPS`) column
hr = df['HR'] * 4
triple = df['3B'] * 3
double = df['2B'] * 2
df['OPS'] = df['OBP'] + df['Slug_Percent']

#create fantasy value
df['fantasy_value'] = df['single'] + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] *4) + df['R'] + df['RBI'] + df['BB'] + (df['SB'] * 2) + df['HBP'] - df['CS'] - (df['SO'] / 2)

df.sort_values(by=['playerID', 'yearID'], inplace=True)
df = df.loc[df['AB'] > 100]

df['next_id'] = df['playerID'].shift(-1)
df['new_fantasy'] = df['fantasy_value'].shift(-1)
for i, row in df.iterrows():
    if row['playerID'] != row['next_id']:
        df.at[i, 'new_fantasy'] = row['fantasy_value']

# Check for null values
print(df.isnull().sum(axis=0).tolist())

# Eliminate rows with null values
df = df.dropna()

#df.to_csv('testExample.csv', sep=",")

#shuffle data
df = df.reindex(
    np.random.permutation(df.index)
)
print("filter df for players up to 2017")
# Filter `df` for players so only up to 2017
data = df.loc[df['yearID'] <= 2017]

print("length: -----------")
print(len(data))

print("filter df for only 2018 players")
# Filter `df` for year 2018 to use on test data
test_data = df.loc[df['yearID'] == 2018]


#test_data.info()
#print(test_data)



def preprocess_features(df):

  selected_features = df[['AVE', 'OBP', 'Slug_Percent', 'OPS', 'G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'HBP']]

  processed_features = selected_features.copy()

  return processed_features

def preprocess_targets(df):

  output_targets = pd.DataFrame()
  
  #output_targets["fantasy_value"] = (df['single'] + (df['2B'] * 2) + (df['3B'] * 3) + (df['HR'] *4) + df['R'] + \
   #df['RBI'] + df['BB'] + (df['SB'] * 2) + df['HBP'] - df['CS'] - (df['SO'] / 2)) * 2

  output_targets["new_fantasy"] = df["new_fantasy"]
  return output_targets


# Choose the first 15000 (out of 21000) examples for training.
training_examples = preprocess_features(data.head(15000))
training_targets = preprocess_targets(data.head(15000))

# Choose the last 5000 (out of 21000) examples for validation.
validation_examples = preprocess_features(data.tail(5000))
validation_targets = preprocess_targets(data.tail(5000))

# Double-check that we've done the right thing.
print("Training examples summary:")
display.display(training_examples.describe())
print("Validation examples summary:")
display.display(validation_examples.describe())

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())


def construct_feature_columns(input_features):

  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_nn_regression_model(
        learning_rate,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):

    periods = 10
    steps_per_period = steps / periods

    # Create a DNNRegressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
        feature_columns=construct_feature_columns(training_examples),
        hidden_units=hidden_units,
        optimizer=my_optimizer,
    )

    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets["new_fantasy"],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets["new_fantasy"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["new_fantasy"],
                                                      num_epochs=1,
                                                      shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        dnn_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()

    print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

    return dnn_regressor

dnn_regressor = train_nn_regression_model(
    learning_rate=0.0004,
    steps=2000,
    batch_size=200,
    hidden_units=[10, 5],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)



test_examples = preprocess_features(test_data)
test_targets = preprocess_targets(test_data)

predict_testing_input_fn = lambda: my_input_fn(test_examples,
                                               test_targets["new_fantasy"],
                                               num_epochs=1,
                                               shuffle=False)

test_predictions = dnn_regressor.predict(input_fn=predict_testing_input_fn)


test_predictions = np.array([item['predictions'][0] for item in test_predictions])

root_mean_squared_error = math.sqrt(
    metrics.mean_squared_error(test_predictions, test_targets))

print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)


final_dict = {}

for i in range(len(test_data)):
    final_dict[i] = test_predictions[i]

print(final_dict)
sorted_dict = sorted(final_dict.keys(), key=lambda x: final_dict[x], reverse=True)

for key in sorted_dict[0:20]:
    print(test_predictions[key])
    print(test_data.iloc[key, 22:24])

sorted_test = test_data.sort_values(by='fantasy_value', ascending=False)

print("print actual values: ")
for i in range(20):
    print(sorted_test.iloc[i])

