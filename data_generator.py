import keras
import h5py
import numpy as np
from keras.utils import to_categorical

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(32,32,32), n_channels=1,
                 num_classes=None, data_file=None,shuffle=True,debug=False,
                 key_x=None, key_y=None):

        'Initialization'
        if data_file is None:
            raise(ValueError("data_file is None"))
        if num_classes is None:
            raise(ValueError("num_classes is None"))
        if key_x is None:
            raise(ValueError("num_classes is None"))
        if key_y is None:
            raise(ValueError("num_classes is None"))
        self.data_file = data_file
        self.data = h5py.File(self.data_file,'r')
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.debug=debug
        self.key_x = key_x
        self.key_y = key_y
        f = h5py.File(self.data_file,'r')
        self.features = f[self.key_x]
        self.labels = f[self.key_y]
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim),dtype='float32')
        y = np.empty((self.batch_size,self.num_classes,), dtype='float32')
        # Generate data
        for i, index in enumerate(indexes):

            X[i,] = self.features[index]
            # Store class
            y[i] = self.labels[index]
        return X/255. ,y/1.

