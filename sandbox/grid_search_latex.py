import pandas as pd

results = pd.read_csv("..\grid_search_results.csv")
pd.set_option('display.max_colwidth', 20)

sorted = results.sort_values(by=['mean_test_score'], ascending=False)
# print(sorted)

print(sorted.iloc[18, 10])

# results = pd.read_csv("..\grid_search_results_temp.csv")
# pd.set_option('display.max_colwidth', 20)
#
# sorted = results.sort_values(by=['mean_test_score'], ascending=False)
# # print(sorted)
#
# print(sorted.iloc[0, 10])
# # exit(0)
#
# cropped1 = sorted.iloc[0:20, [3] + [4] + [5] + [-1] + [-2]]
# cropped1.to_latex("grid_search_results_latex1_temp.txt")
# print(cropped1)
#
# pd.set_option('display.max_colwidth', 16)
# cropped2 = sorted.iloc[0:20, 6:10]# + [-1] + [-2]]
# # cropped2 = sorted.iloc[0:20, [6] + [6] + [7] + [8] + [9] + [10]]# + [-1] + [-2]]
# cropped2.to_latex("grid_search_results_latex2_temp.txt", index=False)
# print(cropped2)
