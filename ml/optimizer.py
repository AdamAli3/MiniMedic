import hypothesis as H
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

max_iters = 1500
min_iters = 1500
last_iters = min_iters

max_alpha = 100
min_alpha = 0.0001
last_alpha = min_alpha

min_error = 100

test = 0

df = pd.read_csv("column_3C_weka.csv")
df2 = df.copy()

print("Reading saved parameters...")
f = open("parameters.txt", "r")
lines = f.readlines()

min_error_line = lines[3]
min_error = min_error_line[min_error_line.index(":") + 1 : min_error_line.index("%")]
min_error = float(min_error)

f.close()

print("Reading last test...")
f = open("last_test.txt", "r")
lines = f.readlines()

test_line = lines[0]
test = test_line[test_line.index(":") + 1 : test_line.index("\n")]
test = int(test)

last_alpha_line = lines[1]
last_alpha = last_alpha_line[last_alpha_line.index(":") + 1 : last_alpha_line.index("\n")]
last_alpha = float(last_alpha)

last_iters_line = lines[2]
last_iters = last_iters_line[last_iters_line.index(":") + 1 : last_iters_line.index("\n")]
last_iters = int(last_iters)

f.close()

print("Creating testing data...")
h1 = H.hypothesis(["Hernia"], ["Spondylolisthesis","Normal"], [0,0,0,0,0])
h2 = H.hypothesis(["Spondylolisthesis"], ["Hernia","Normal"], [0,0,0,0,0])
h3 = H.hypothesis(["Normal"], ["Spondylolisthesis","Hernia"], [0,0,0,0,0])

df2.loc[df2.out == "Hernia", 'out'] = 0
df2.loc[df2.out == "Spondylolisthesis", 'out'] = 1
df2.loc[df2.out == "Normal", 'out'] = 2

testing_input = df2[df2.out == 0].iloc[0:40, :6]
testing_input = testing_input.append(df2[df2.out == 1].iloc[0:40, :6], ignore_index=True)
testing_input = testing_input.append(df2[df2.out == 2].iloc[0:40, :6], ignore_index=True)

expected_output = df2[df2.out == 0].iloc[0:40, 6]
expected_output = expected_output.append(df2[df2.out == 1].iloc[0:40, 6], ignore_index=True)
expected_output = expected_output.append(df2[df2.out == 2].iloc[0:40, 6], ignore_index=True)
expected_output = expected_output.values

alpha = last_alpha
total_tests = (max_alpha/min_alpha)
progress = test/total_tests * 100
print("-----------------------------------------------")
print("TEST: [", test, "-----", str(round(progress, 2)), "%]")
print("ALPHA: [", last_alpha, "]")
print("ITERATIONS: [", last_iters, "]")
print("MINIMUM: [", str(round(min_error, 2)), "%]")
run = input("Run with the following parameters? (y/n): ")

if run == "n":
    quit()

while alpha <= max_alpha:
    iters = last_iters
    h1 = H.hypothesis(["Hernia"], ["Spondylolisthesis","Normal"], [0,0,0,0,0])
    h2 = H.hypothesis(["Spondylolisthesis"], ["Hernia","Normal"], [0,0,0,0,0])
    h3 = H.hypothesis(["Normal"], ["Spondylolisthesis","Hernia"], [0,0,0,0,0])
    while iters <= max_iters:
        test += 1
        print("-----------------------------------------------")
        progress = test/total_tests * 100
        print("TEST: [", test, "-----", str(round(progress, 2)), "%]")
        print("ALPHA: [", alpha, "]")
        print("ITERATIONS: [", iters, "]")
        print("MINIMUM: [", str(round(min_error, 2)), "%]")

        print("Training data model...")
        h1 = H.hypothesis(["Hernia"], ["Spondylolisthesis","Normal"], h1.theta)
        h2 = H.hypothesis(["Spondylolisthesis"], ["Hernia","Normal"], h2.theta)
        h3 = H.hypothesis(["Normal"], ["Spondylolisthesis","Hernia"], h3.theta)
        h1.train(df, alpha, min_iters)
        h2.train(df, alpha, min_iters)
        h3.train(df, alpha, min_iters)

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
            f = open("parameters.txt", "w+")
            f.write("TEST:" + str(test) + "\n")
            f.write("ALPHA:" + str(alpha) + "\n")
            f.write("ITERATIONS:" + str(iters) + "\n")
            f.write("ERROR:" + str(round(avg_error, 2)) + "%" + "\n")
            f.close()


        f = open("last_test.txt", "w+")
        f.write("TEST:" + str(test) + "\n")
        f.write("ALPHA:" + str(alpha) + "\n")
        f.write("ITERATIONS:" + str(iters) + "\n")
        f.write("ERROR:" + str(round(avg_error, 2)) + "%" + "\n")
        f.close()
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
