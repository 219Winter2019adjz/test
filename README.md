# ECE219 Project 1 - Classification Analysis on Textual Data

## Necessary packages

The python packages required to run this project are given in `requirements.txt`.

The Natural Language Processing Toolkit (NLTK) package requires extra data in order to be used. In order to
download the necessary packages, run the following:
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

## Running the code

In order to use this project, run `proj1.ipynb` by starting a Jupyter Notebook session and stepping through each block sequentially.

Some sections require that previous sections be run before them. The section concerning Question 7 (Grid Search) is the only exception.

## Notes

Question 7 takes an arduous amount of time to run. The results for Question 7 are therefore saved to a csv file, if you don't wish to run the grid search, then load the previous csv by running the block following the grid search instead.
