import hypothesis as H
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("Extracting data points...")
df = pd.read_csv("column_3C_weka.csv")
df2 = df.copy()

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

# print(predict)


# h1.plot_data(1);
# h2.plot_data(2);
# h3.plot_data(3);
