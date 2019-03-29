import hypothesis as H
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

max_iters = 4000
min_iters = 100

max_alpha = 100
min_alpha = 0.0001

min_error = 100

df = pd.read_csv("column_3C_weka.csv")
df2 = df.copy()

print("Creating testing data...")
h1 = H.hypothesis(["Hernia"], ["Spondylolisthesis","Normal"], [0,0,0,0,0])
h2 = H.hypothesis(["Spondylolisthesis"], ["Hernia","Normal"], [0,0,0,0,0])
h3 = H.hypothesis(["Normal"], ["Spondylolisthesis","Hernia"], [0,0,0,0,0])

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

alpha = min_alpha
test = 0
total_tests = ((max_iters - min_iters)/100) * ((max_alpha - min_alpha)/min_alpha)
while alpha <= max_alpha:
    iters = min_iters
    while iters <= max_iters:
        test += 1
        print("-----------------------------------------------")
        progress = test/total_tests
        print("TEST: [", test, "-----", str(round(progress, 2)), "%]")
        print("ALPHA: [", alpha, "]")
        print("ITERATIONS: [", iters, "]")
        print("MINIMUM: [", str(round(min_error, 2)), "%]")

        print("Training data model...")
        h1.train(df, alpha, iters)
        h2.train(df, alpha, iters)
        h3.train(df, alpha, iters)

        print("Predicting outcomes...")
        predicted_output = []
        for index, row in testing_input.iterrows():
            x_array = np.array(row)
            predict = [h1.predict(x_array), h2.predict(x_array), h3.predict(x_array)]
            predicted_output.append(np.argmax(predict))

        print("Calculating error")
        sum_error = 0
        for i in range(len(expected_output)):
            if predicted_output[i] != expected_output[i]:
                error = 1
            else:
                error = 0
            sum_error += error

        avg_error = sum_error/len(predicted_output) * 100
        print("Average Error: " + str(round(avg_error, 2)) + "%")
        if avg_error < min_error:
            min_error = avg_error
            print("NEW MINIMUM")
            save_alpha = alpha
            save_iters = iters
            save_test = test

        print("-----------------------------------------------")
        iters += min_iters
    alpha += min_alpha

print("-----------------------------------------------")
print("-----------------------------------------------")
print("-----------------------------------------------")
print("TEST: [", save_test, "]")
print("ALPHA: [", save_alpha, "]")
print("ITERATIONS: [", save_iters, "]")
print("ERROR: [", min_error, "]")
print("-----------------------------------------------")
print("-----------------------------------------------")
print("-----------------------------------------------")
