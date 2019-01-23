import pandas as pd

results = pd.read_csv("..\\best_multi_nb.csv")
pd.set_option('display.max_colwidth', 20)

sorted = results.sort_values(by=['mean_test_score'], ascending=False)
# print(sorted)

print(sorted.iloc[0, 10])

cropped1 = sorted.iloc[0:5, [3] + [4] + [5] + [-1] + [-2]]
print(cropped1)

pd.set_option('display.max_colwidth', 16)
cropped2 = sorted.iloc[0:5, 6:10]# + [-1] + [-2]]
print(cropped2)
