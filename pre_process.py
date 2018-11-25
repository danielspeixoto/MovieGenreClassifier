import csv
import pandas as pd
import pickle
from sklearn.utils import resample
import data

print("Reading")
df, y = data.get("movies.csv", ['comedy', 'drama'])
# df.apply(lambda doc: pre_process(doc))
print("Processing...")
pre_data = [data.pre_process(row) for idx, row in df.iteritems()]

df = pd.DataFrame({
    "Plot": pre_data,
    "Genre": y
})

print("Original proportion: ")
print(df['Genre'].value_counts())

# Down sampling
majority = df[df['Genre'] == 1]
minority = df[df['Genre'] == 0]

downsampled = resample(majority, replace=False, n_samples=len(minority))
df = pd.concat([downsampled, minority])

print("New proportion: ")
print(df['Genre'].value_counts())

print("Saving")
df.to_csv('movies-pre_processed.csv')
# with open('pre_processed.csv', 'w', newline='\n') as file:
#     wr = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
#     wr.writerow(["Plot", "Genre"])
#     for item in df:
#         wr.writerow(item)
