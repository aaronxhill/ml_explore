# ml_explore

A Python module to support the process of feature extraction and model tuning for machine learning in [`scikit-learn`](http://scikit-learn.org/). The methods in this module aid the exploration and understanding of input and output. Visualize the distributions of features in large feature sets. Quickly calculate various model performance measures on training and test sets. In natural language processing feature sets, view the original raw text input data for correct and incorrect classifications. [Contributions](https://github.com/visualizedata/github-workflow) are welcomed and encouraged. 

### Classes:

## `BinaryClassificationPerformance()`

Description of class

#### Parameters:

`__init__(predictions, labels, desc, probabilities=None)`

**`predictions`** : ndarray
> ndarray of shape(n, 0) and data type boolean, containing predicted values for Y

**`labels`** : ndarray
> ndarray of shape(n, 0) and data type boolean, containing Y labels

**`desc`** : string
> description of instance

**`probabilities`** : ndarray, optional
> ndarray of shape(n, 0) and data type boolean, containing probabilities that Y is equal to `True`

#### Methods:

method | description
--- | ---
`compute_measures`() | Compute machine learning performance measures defined by [Flach](https://www.cs.bris.ac.uk/~flach/mlbook/) p. 57  
`img_indices`() | Get the indices of true and false positives to be able to locate the corresponding images in a list of image names

#### Attributes:

attribute | description
--- | ---
`probabilities` | `ndarray` of shape(n, 0) of probabilities that Y is equal to True
`performance_df` | `DataFrame` with two columns, one each for predictions and labels
`desc` | `string` description of instance
`performance_measures` | `dict` of performance measures
`image_indices` | `dict` of image indices

#### Example:

```python
prc_performance = BinaryClassificationPerformance(prc.predict(data_train), y_train, 'prc')
prc_performance.compute_measures()
print(prc_performance.performance_measures)
```

```
{'Recall': 0.42857142857142855, 'Precision': 0.81818181818181823, 'Neg': 1179, 'TP': 9, 'Pos': 21, 'FN': 12, 'Accuracy': 0.98833333333333329, 'TN': 1177, 'FP': 2}
```

## `VizColumns()`

Description of class

#### Parameters:

`__init__(X)`

**`X`** : scipy.sparse.csr.csr_matrix
> sparse matrix

#### Methods:

method | description
--- | ---
`column_means_distribution`() | View histogram of column means
#### Attributes:

attribute | description
--- | ---
`X` | original sparse matrix
`X_csr_means` | `list` of means for the columns in X

#### Example:

```python
vc = VizColumns(X_hv)
vc.column_means_distribution()
```

```
Distribution of means of 131072 columns in X.
```
![histogram]('https://github.com/aaronxhill/ml_explore/raw/master/img/hist_viz_columns.png')