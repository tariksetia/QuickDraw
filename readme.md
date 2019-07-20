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


### MLP Observation
- Performed `GridSearchCV` for singe layer with `units` = `[100, 512, 784]` and `learning_rate` = `[0.01, 0.0001, 0.001]`
- Hidden layer with 100 `units` gave best accuracy but as the number of samples doubled, 784 `unit` with `lr` =  0.0001 gave the best result and the previous configuration over fitted. 
- `lr=0.001` and `units=784` gave best score only once but that was very very close to the score with `lr=0.00001`. So I chose `lr=0.0001` without any further experiments
- Performed `GridSearchCV` again but this time with two  hidden layers with configs = `[(100,100,), (512,512), (784,784,) and (784,)(previous best)]` and `learning_rate` = `[ 0.0001, 0.001]`. Single Hidden layer with with `units=784` and `lr=0.0001` gave the best score.
- Mean score in above cases was around  0.16. But I continued with `units=784` and `lr=0.0001`
- `Epochs` used for `GridSearchCV` were 20 and no `batch size` was specified
- Next, I trained `MLP(units=784 and learning_rate=0.0001)` for 10 `epochs` with  following number of samples:
    - Test Set = 921706
    - Validation Set = 230421
    - Test Set = 288027
- First it was trained with all the data in memomry with `batch_size` of 32. It seem the was overfitting as validation loss kept increasing with each epoch

- Next I retrained the model for 10 epochs but this time with `L2 regularization` and  `lambda=0.001`
- Keras evaluation score is [0.4670979449351629, 0.8884479166666667]
 and curve color is green

#### - Loss Curves (Loss vs samples)
! [./curves/mlp/Training Curves.png]


