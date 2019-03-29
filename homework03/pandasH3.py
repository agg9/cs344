'''
    Homework03 for cs344 at Calvin College

    @author Austin Gibson
    @version March 27, 2019
'''

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
import numpy as np
import pandas as pd

from keras.datasets import boston_housing

#(train_images, train_labels), (test_images, test_labels) = boston_housing.load_data()

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

boston_housing_dataframe = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")

#shuffle data
boston_housing_dataframe = boston_housing_dataframe.reindex(
    np.random.permutation(boston_housing_dataframe.index))

# 2a.  Compute dimensions, and print values
def print_structure_dimensions():
    """
    print(
        'training data (axes: {}; shape: {}; type: {}): \n{}\n\n'.format(train_labels.ndim, train_labels.shape,
                                                                         train_labels.dtype, train_labels)
    )
    print(
        'testing data (axes: {}; shape: {}; type: {}): \n{}\n\n'.format(test_labels.ndim, test_labels.shape,
                                                                         test_labels.dtype, test_labels)
    )
    """
    print('boston housing dataframe')
    print('size: {}'.format(boston_housing_dataframe.size))
    print('rows: {}'.format(len(boston_housing_dataframe)))
    columns = boston_housing_dataframe.size / len(boston_housing_dataframe)
    print('columns: {} '.format(columns))
print_structure_dimensions()


# 2b.  Construct suitable testing set, training set, and validation set for data
def preprocess_features(boston_housing_dataframe):
  """Prepares input features from Boston Housing Dataset.

  Args:
    boston_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  selected_features = boston_housing_dataframe[
    ["crim",
     "zn",
     "indus",
     "chas",
     "nox",
     "rm",
     "age",
     "dis",
     "rad",
     "tax",
     "ptratio",
     "b",
     "lstat",
     "medv"]]
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  return processed_features

def preprocess_targets(boston_housing_dataframe):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    boston_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["medv"] = (
    boston_housing_dataframe["medv"] / 1000.0)
  return output_targets


#create training and validation sets.  Dataset has 506 rows.  Training/Validation/Testing 80/10/10 split
training_examples = preprocess_features(boston_housing_dataframe.head(404))
training_examples.describe()

training_targets = preprocess_targets(boston_housing_dataframe.head(404))
training_targets.describe()

temp_dataframe = boston_housing_dataframe.tail(102)
validation_examples = preprocess_features(temp_dataframe.head(56))
validation_examples.describe()

validation_targets = preprocess_targets(temp_dataframe.head(56))
validation_targets.describe()

testing_examples = preprocess_features(temp_dataframe.tail(56))
testing_examples.describe()

testing_targets = preprocess_targets(temp_dataframe.tail(56))
testing_targets.describe()


#2c.  Create a synthetic feature.
boston_housing_dataframe["lower_pop_rooms"] = boston_housing_dataframe["rm"] * boston_housing_dataframe["lstat"]


