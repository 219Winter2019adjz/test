# ########################################################################################################################
# # Fetching 20NewsGroups dataset
# from sklearn.datasets import fetch_20newsgroups
# # Refer to the offcial document of scikit-learn for detailed usages:
# # http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
# categories = ['comp.graphics', 'comp.sys.mac.hardware']
# twenty_train = fetch_20newsgroups(subset='train', # choose which subset of the dataset to use; can be 'train', 'test', 'all'
#                                   categories=categories, # choose the categories to load; if is `None`, load all categories
#                                   shuffle=True,
#                                   random_state=42, # set the seed of random number generator when shuffling to make the outcome repeatable across different runs
#                                   # remove=['headers'],
#                                   )
# twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
#
# ########################################################################################################################
# # Convert train and test data to counts
#
# from sklearn.feature_extraction.text import CountVectorizer
# count_vect = CountVectorizer(min_df=3, stop_words='english')
#
# # do for training
# X_train_counts = count_vect.fit_transform(twenty_train.data)
#
# # do for testing
# X_test_counts = count_vect.transform(twenty_test.data)
#
# ########################################################################################################################
# # Get TFIDF of training and test sets
#
# from sklearn.feature_extraction.text import TfidfTransformer
# tfidf_transformer = TfidfTransformer()
#
# # do for training
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#
# # do for testing
# X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)


########################################################################################################################
# Fetching 20NewsGroups dataset
from sklearn.datasets import fetch_20newsgroups
# Refer to the offcial document of scikit-learn for detailed usages:
# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
categories = ['comp.graphics', 'comp.sys.mac.hardware']
twenty_train = fetch_20newsgroups(subset='train', # choose which subset of the dataset to use; can be 'train', 'test', 'all'
                                  categories=categories, # choose the categories to load; if is `None`, load all categories
                                  shuffle=True,
                                  random_state=42, # set the seed of random number generator when shuffling to make the outcome repeatable across different runs
                                  remove=('headers',),
                                  )
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

########################################################################################################################
# Train a Naive Bayes Gaussian classifier on the TFIDF training set from problem 2

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB().fit(twenty_train.data, twenty_train.target)

########################################################################################################################
# Generate predictions for test set

predicted = clf.predict(twenty_test)

for i, category in enumerate(predicted):
    if i < 5:
        print('{} =? {}'.format(twenty_test.target_names[category], twenty_test.target_names[twenty_test.target[i]]))
    else:
        break
