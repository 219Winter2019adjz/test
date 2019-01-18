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
                                  # remove=['headers'],
                                  )
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

########################################################################################################################
# Convert train and test data to counts

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(min_df=3, stop_words='english')

# do for training
X_train_counts = count_vect.fit_transform(twenty_train.data)

# do for testing
X_test_counts = count_vect.transform(twenty_test.data)

########################################################################################################################
# Get TFIDF of training and test sets

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()

# do for training
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# do for testing
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

########################################################################################################################
# Perform NMF

from sklearn.decomposition import NMF
model = NMF(n_components=50, init='random', random_state=42)

# do for training
W_nmf_train_reduced = model.fit_transform(X_train_tfidf)
H_nmf_train_reduced = model.components_

print(W_nmf_train_reduced.shape)
print(twenty_train.target.shape)

# do for testing
W_nmf_test_reduced = model.transform(X_test_tfidf)
H_nmf_test_reduced = model.components_

########################################################################################################################
# Train a Naive Bayes Gaussian classifier on the TFIDF training set from problem 2

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB().fit(W_nmf_train_reduced, twenty_train.target)

########################################################################################################################
# Generate predictions for test set

predicted = clf.predict(W_nmf_train_reduced)
correct = 0
for i, category in enumerate(predicted):
    if category == twenty_train.target[i]:
        correct += 1
    # if i < 5:
    #     print('{} =? {}'.format(twenty_test.target_names[category], twenty_test.target_names[twenty_test.target[i]]))
    # else:
    #     break
print('Accuracy of NB Gaussian (train): {}'.format(correct / W_nmf_train_reduced.shape[0]))

predicted = clf.predict(W_nmf_test_reduced)
correct = 0
for i, category in enumerate(predicted):
    if category == twenty_test.target[i]:
        correct += 1
    # if i < 5:
    #     print('{} =? {}'.format(twenty_test.target_names[category], twenty_test.target_names[twenty_test.target[i]]))
    # else:
    #     break
print('Accuracy of NB Gaussian (test): {}'.format(correct / W_nmf_test_reduced.shape[0]))
