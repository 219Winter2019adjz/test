import pandas as pd

results = pd.read_csv("grid_search_results_copy.csv")
pd.set_option('display.max_colwidth', 20)

sorted = results.sort_values(by=['mean_test_score'], ascending=False)
print(sorted)

cropped1 = sorted.iloc[0:20, [3] + [4] + [5] + [-1] + [-2]]
cropped1.to_latex("grid_search_results_latex1.txt")
print(cropped1)

cropped2 = sorted.iloc[0:20, [5] + [6] + [7] + [8]]# + [-1] + [-2]]
cropped2.to_latex("grid_search_results_latex2.txt", index=False)
print(cropped2)
