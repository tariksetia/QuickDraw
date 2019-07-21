### `QuickDrawDataset`
##### File: `quick_draw_data.py`

QuickDrawDataset takes multiple ndjson files as input and creates a sigle `data.HDF5` file with folllowing datasets:
- `X_train`, `y_train`    Training features and labels
- `X_valid`, `y_valid`    Validation features and labels
- `X_test`, `y_test`      Test features and labels

`ndjosn` files are downloaded for google bucket and contains data for drawing strokes. `QuickDrawDataset` convert these stokes to `28 by 28 np.array` features.

`QuickDrawDataset.split` returns train,validation and test batch generators. Each iteration on generator returns X and y of shape
`(batch_size, 28, 28)` and `(batch_size, 3)`


</br>

### `DataGenerator`
##### File: `data_generator.py`
This class inherits from `keras.utils.sequence`. It actually creates the `train`, `validation` and `test` batch generators, mentions above

https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
</br>


### `QuickDrawModel`
##### File: `model.py`
It is a little abstaction over keras. It takes multiple `MLP architecture`  as input. After building  `keras` model for  all the input architectures, it trains them and saves training history and evalutions in `QuickDrawModel.histories` and `QuickDrawModel.evaluations`.

These variables are later used in notebook `quick_draw.ipynb` for visualizing training

It also save Keras model in `'./model/` as HDF5 files and training_history for each architecture in a single `JSON` file `./history.json`


</br>




