from root.utils import * # pylint: disable= unused-wildcard-import

from root.readData import readCSV

df, test_df = readCSV(20, test_size=0.1)


df.to_csv('../data/output/csv/train0.csv')
test_df.to_csv('../data/output/csv/test0.csv')


print(len(df[df['target']==0].index))
print(len(df[df['target']==1].index))
print(len(test_df[test_df['target']==0].index))
print(len(test_df[test_df['target']==1].index))
