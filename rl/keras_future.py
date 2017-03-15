import keras
import keras.layers
import keras.models


def concatenate(x):
    if hasattr(keras.layers, 'Concatenate'):
        return keras.layers.Concatenate()(x)
    else:
        return keras.layers.merge(x, mode='concat')


def add(x):
    if hasattr(keras.layers, 'Add'):
        return keras.layers.Add()(x)
    else:
        return keras.layers.merge(x, mode='sum')


def Model(input, output, **kwargs):
    if int(keras.__version__.split('.')[0]) >= 2:
        return keras.models.Model(inputs=input, outputs=output, **kwargs)
    else:
        return keras.models.Model(input=input, output=output, **kwargs)
