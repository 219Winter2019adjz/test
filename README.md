# ECE219 Project 1 - Classification Analysis on Textual Data

## Necessary packages

The python packages required to run this project are given in `requirements.txt`.

The Natural Language Processing Toolkit (NLTK) package requires extra data for them to be used. To
download the necessary packages, run the following:
```python
import nltk
nltk.download('punkt')
```

## Running the code

In order to use this project, run `proj1.ipynb` by starting a Jupyter Notebook session and stepping through each block sequentially.

Some sections require that previous sections be run before them. The section concerning Question 7 (Grid Search) is the only exception.

## Notes

Question 7 takes an arduous amount of time to run. The results for Question 7 are therefore saved to a csv file, if you don't wish to run the grid search, then load `grid_search_results.csv` by running the block following the grid search instead. More specifially, the following code snippet:
```python
results = pd.read_csv('grid_search_results.csv')
results
```

can be used instead of redoing the grid search:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD, NMF

# used to cache results
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory
# print(__doc__)
cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=10)

pipeline = Pipeline([
    ('import', Importer(remove=None)),
    ('vect', CountVectorizer(min_df=1, stop_words='english', analyzer='word')),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', TruncatedSVD()),
    ('clf', GaussianNB()),
],
memory=memory
)

param_grid = [
    {
        'import__remove': [None, ['headers', 'footers']],
        'vect__min_df': [3, 5],
        'vect__analyzer': ['word', stem_rmv_punc],
        'reduce_dim': [TruncatedSVD(random_state=42, n_components=50), NMF(random_state=42, n_components=50)],
        'clf': [LinearSVC(C=10,max_iter=100000),
               LogisticRegression(random_state=42, C=0.1, penalty='l1', solver='liblinear'),
               LogisticRegression(random_state=42, C=0.01, penalty='l2', solver='liblinear'),
               GaussianNB()]
    }
]

grid = GridSearchCV(pipeline, cv=5, n_jobs=1, param_grid=param_grid, scoring='accuracy')
grid.fit(twenty_train.data, twenty_train_binary_labels)
rmtree(cachedir)
```
