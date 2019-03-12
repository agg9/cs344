'''
 @author austin gibson
 @version march 11, 2019

'''

import numpy as np
from keras.datasets import boston_housing

(train_images, train_labels), (test_images, test_labels) = boston_housing.load_data()
def print_structures():
    print(
        f'6.2.i \
                \n\tcount: {len(train_images)} \
                \n\tcount: {len(test_labels)}\n',

        f'6.2.ii training \
                \n\tdimensions: {train_labels.ndim} \
                \n\tshape: {train_labels.shape} \
                \n\tdata type: {train_labels.dtype} \
                \n\tvalues: {train_labels}\n',
        f'6.2.ii testing \
                    \n\tdimensions: {test_labels.ndim} \
                    \n\tshape: {test_labels.shape} \
                    \n\tdata type: {test_labels.dtype} \
                    \n\tvalues: {test_labels}\n',
    )

print_structures()