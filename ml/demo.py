import hypothesis as H
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

print("Extracting data points...")
df = pd.read_csv("column_3C_weka.csv")
df2 = df.copy()

try:
    print("Looking for pickles...")
    h1 = pickle.load(open("h1.p", "rb"))
    h2 = pickle.load(open("h2.p", "rb"))
    h3 = pickle.load(open("h3.p", "rb"))
    print("Pickles found!")
except FileNotFoundError:
    print("No pickles found!")
    print("Training data model...")
    h1 = H.hypothesis(["Hernia"], ["Spondylolisthesis","Normal"], [0,0,0,0,0])
    h2 = H.hypothesis(["Spondylolisthesis"], ["Hernia","Normal"], [0,0,0,0,0])
    h3 = H.hypothesis(["Normal"], ["Spondylolisthesis","Hernia"], [0,0,0,0,0])
    h1.train(df)
    h2.train(df)
    h3.train(df)

print("Creating testing data...")
df2.loc[df2.out == "Hernia", 'out'] = 0
df2.loc[df2.out == "Spondylolisthesis", 'out'] = 1
df2.loc[df2.out == "Normal", 'out'] = 2

testing_input = df2[df2.out == 0].iloc[0:40, :6]
testing_input = testing_input.append(df2[df2.out == 1].iloc[61:161, :6], ignore_index=True)
testing_input = testing_input.append(df2[df2.out == 2].iloc[211:277, :6], ignore_index=True)

expected_output = df2[df2.out == 0].iloc[0:40, 6]
expected_output = expected_output.append(df2[df2.out == 1].iloc[61:161, 6], ignore_index=True)
expected_output = expected_output.append(df2[df2.out == 2].iloc[211:277, 6], ignore_index=True)
expected_output = expected_output.values


print("Predicting outcomes...")
predicted_output = []

for index, row in testing_input.iterrows():
    x_array = np.array(row)
    predict = [h1.predict(x_array), h2.predict(x_array), h3.predict(x_array)]
    predicted_output.append(np.argmax(predict))


print("Calculating error")
sum_error = 0
for i in range(len(expected_output)):
    error = np.abs(predicted_output[i] - expected_output[i])
    sum_error += error


avg_error = sum_error/len(predicted_output)

print("Average Error: " + str(round(100 * avg_error, 2)) + "%")

print("Genereating pickles...")
pickle.dump(h1, open("h1.p", "wb"))
pickle.dump(h2, open("h2.p", "wb"))
pickle.dump(h3, open("h3.p", "wb"))
print("Hypothesis saved")
