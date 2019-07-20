import os
import json
import shutil
from glob import glob
from random import shuffle
from math import ceil
from multiprocessing import Pool

import h5py
import keras
import numpy as np
from PIL import Image, ImageDraw
from keras.utils import to_categorical
from data_generator import DataGenerator

class QuickDrawDataset:
    def __init__ (self,num_classes=None, workers=1, path=None, output_file_name='data.hdf5', samples_per_class=None,
                     color_output=False, out_shape=(28,28),split_spot=80, save_hdf5=False,split=80, batch_size=32, debug=True):
        
        if num_classes is None:
            raise ValueError("num_class is None")
        if path is None:
            raise ValueError("Please Provide .NDJSON Path")
        filexp = path + '/*.ndjson'  
        if len(glob(filexp)) == 0:
            raise Exception("No ndjson file found  " + filexp) 
        
        self.num_classes = num_classes
        self.workers = workers
        self.path = path
        self.output = output_file_name
        self.split_spot = split_spot
        self.color_output = color_output
        self.hdf5 = self.path + '/hdf5'
        self.file_to_class = None
        self.samples_per_class = samples_per_class
        self.debug = debug
        self.files = glob(self.path + '/*.ndjson')
        self.out_shape = out_shape
        self.batch_size = batch_size
        self.output_file = self.path + '/final/data.hdf5'
        self.create()
    
    def clear(self):
        os.unlink(self.output_file)

    
    def create(self):
        if not os.path.isfile(self.output_file):
            self._strokes_to_image()
            self._merge_hdf5()
            self.clear_files()
        else:
            print("Warning: Using exiting data file @ ", self.output_file)
    

    def _strokes_to_image(self):
        #raise error if NDJSON dir is empty or cum_class doesnt match numbero of nd json file
        files = glob(self.path + '/*.ndjson')
        if len(files) != self.num_classes:
            raise Exception("{} files for porcessing {} classes. Expecting m files for m classes".format(len(files),self.num_class))
        
        # if dir HDF5 does not exist, make it
        os.mkdir(self.hdf5) if not os.path.isdir(self.hdf5) else None
        
        #Fan out processing for coverting strokes to drawings asn save them respective dirs
        self._fanout(self.process_file,files)
    

    def _merge_hdf5(self):
        files = glob(self.hdf5 + '/*.hdf5')
        output_dir = self.path + '/final'
        output_file = output_dir + '/data.hdf5'

        if os.path.isfile(output_file):
            print("Warning: Using existing data file @ " + output_file)
            return
        
        if  not os.path.isdir(output_dir): 
            os.mkdir(output_dir)

        file_ = files.pop(0)
        if self.debug: 
            print("Reading {}".format(file_))
        
        f = h5py.File(file_,'r')
        X_train, X_valid, X_test = f['X_train'][()], f['X_valid'][()], f['X_test'][()]
        y_train, y_valid, y_test  = f['y_train'][()], f['y_valid'][()], f['y_test'][()]
        f.close()

        for file_ in files:
            if self.debug: print("Reading {}".format(file_))
            with h5py.File(file_, 'r') as f:
                X_train = np.concatenate([X_train,f['X_train'][:]], axis=0)
                X_valid = np.concatenate([X_valid, f['X_valid'][()]], axis=0)
                X_test = np.concatenate([X_test, f['X_test'][()]], axis=0)
                y_train = np.concatenate([y_train, f['y_train'][()]], axis=0)
                y_valid = np.concatenate([y_valid, f['y_valid'][()]], axis=0)
                y_test = np.concatenate([y_test, f['y_test'][()]], axis=0)
        
        if self.debug:
            print("Merging all Features into {}".format(output_file))
        
        try:
            with h5py.File(output_file, 'w') as f:
                f.create_dataset("X_train", shape=(len(X_train), *self.out_shape), data=X_train)
                f.create_dataset("X_valid", shape=(X_valid.shape[0], *self.out_shape), data=X_valid)
                f.create_dataset("X_test", shape=(X_test.shape[0], *self.out_shape), data=X_test)
                f.create_dataset("y_train", shape=(y_train.shape[0], self.num_classes), data=y_train)
                f.create_dataset("y_valid", shape=(y_valid.shape[0], self.num_classes), data=y_valid)
                f.create_dataset("y_test", shape=(y_test.shape[0], self.num_classes), data=y_test)
        except Exception as e:
            os.unlink(output_file)
            raise(e)
        self.output_file = output_file

    @property
    def num_samples(self):
        output_dir = self.path + '/final'
        output_file = output_dir + '/data.hdf5'
        
        if os.path.isfile(output_file):
            with h5py.File(output_file, 'r') as f:
                return sum((len(f['X_train']), len(f['X_valid']), len(f['X_test'])))
        
        files = glob(self.hdf5 + '/*.hdf5')
        if len(files) == 0:
            raise Exception("HDF5 folder is empty")
        
        samples = 0
        for f in files:
            with h5py.File(f, 'r') as data:
                samples += sum(len(f['X_train']), len(f['X_valid'], len(f['X_test'])))
        return samples
    
    @property
    def split_size(self):
        if os.path.isfile(self.output_file):
            with h5py.File(self.output_file, 'r') as f:
                return (len(f['X_train']), len(f['X_valid']), len(f['X_test']))
        else:
            return(None,None,None)

    @property
    def mapping(self):
        files = [file.split('/')[-1].split('.')[0].lower() for file in self.files]
        files = sorted(files)
        return {
            "classes": {file:i for i,file in enumerate(files)},
            "targets": {str(to_categorical(i, num_classes=self.num_classes, dtype='float32')): file for i,file in enumerate(files)}
        }  
    
    @staticmethod    
    def convert_drawing_to_array_gray(drawing, new_shape):
        im = Image.new('L', (255, 255),color="black")
        draw = ImageDraw.Draw(im)
        for num, i in enumerate(drawing):
            try:
                points = list(zip(i[0],i[1]))
            except Exception as e:
                print(drawing)
            draw.line(points,fill="white", width=5)

        resized = im.resize(new_shape, Image.ADAPTIVE)
        return resized

    def process_file(self, file):
        if self.debug: print("Processing strokes started for: ", file)
        file_name = file.split('/')[-1].split('.')[0].lower()
        mapping = self.mapping
        if os.path.isfile(self.hdf5 + '/' + file_name + '.hdf5'):
            if self.debug: print("HDF5 Image file exist for: " + file)
            return

        drawings = [json.loads(line)['drawing'] for line in open(file)]
        if self.samples_per_class is not None:
            drawings = drawings[:self.samples_per_class]
            
        if self.color_output:
            drawings = [ QuickDrawDataset.convert_drawing_to_array_gray(drawing, self.out_shape) for drawing in drawings ]
        else:
            drawings = [ QuickDrawDataset.convert_drawing_to_array_gray(drawing, self.out_shape) for drawing in drawings ]
            
        drawings = [np.array(drawing, dtype=int) for drawing in drawings]
        drawings = np.array(drawings, dtype=int)
        print(drawings.shape)
        second_split = ceil(len(drawings) *80/100)
        first_split = ceil(second_split *80/100)

        X_train = drawings[:first_split]
        X_valid = drawings[first_split:second_split]
        X_test = drawings[second_split:]
        
        label = to_categorical(mapping["classes"][file_name], num_classes=self.num_classes, dtype=int)
        y_train = np.full(shape=(X_train.shape[0], self.num_classes), fill_value=label)
        y_valid = np.full(shape=(X_valid.shape[0], self.num_classes), fill_value=label)
        y_test = np.full(shape=(X_test.shape[0],self.num_classes ), fill_value=label)
        
        with h5py.File(self.hdf5 + '/' + file_name + '.hdf5', 'w') as f:
            f.create_dataset("X_train", data=X_train, compression="gzip")
            f.create_dataset("X_valid", data=X_valid, compression="gzip")
            f.create_dataset("X_test", data=X_test, compression="gzip")
            f.create_dataset("y_train", data=y_train, compression="gzip")
            f.create_dataset("y_valid", data=y_valid, compression="gzip")
            f.create_dataset("y_test", data=y_test, compression="gzip")

        if self.debug: print("Processing finished for: ", file)
    
    def _fanout(self, function, args):
        pool = Pool(self.workers)
        pool.map(function,args)
    
    def clear_files(self):
        if os.path.isdir(self.hdf5):
            shutil.rmtree(self.hdf5)
    
    def clear_output(self):
        if os.path.isdir(self.path + '/final'):
            os.unlink(self.output_file)
        
    
    @property
    def splits(self):
        ntrain, nvalid, ntest = self.split_size
        train_ids = [i for i in range(ntrain)]
        valid_ids = [i for i in range(nvalid)]
        test_ids = [i for i in range(ntest)]
        
        params = {
            'dim': self.out_shape,
            'batch_size': self.batch_size,
            'num_classes': self.num_classes,
            'n_channels': 1,
            'shuffle': True,
            'data_file': self.output_file
         }
        
        train = DataGenerator(train_ids, key_x="X_train", key_y="y_train", **params)
        valid = DataGenerator(valid_ids, key_x="X_valid", key_y="y_valid", **params)
        test = DataGenerator(test_ids, key_x="X_test", key_y="y_test", **params)

        return train, valid, test
    
    @property
    def full_split(self):
        with h5py.File(self.output_file) as f:
            train  = f['X_train'][:], f['y_train'][:]
            valid = f['X_valid'][:], f['y_valid'][:]
            test = f['X_test'][:], f['y_test'][:]
        
        return train, valid, test

    

if __name__ == '__main__':
    params = {
        "num_classes":3,
        "workers":10,
        "path":"./data",
        "samples_per_class":100
    }

    data = QuickDrawDataset(**params)
    data.create()
    print(data.num_samples)
    f = h5py.File('./data/final/data.hdf5','r')
    print(f['num_samples'][0])
    train, valid, test = data.splits
    for _,y in train:
        print(y)