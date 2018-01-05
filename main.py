import numpy as np
import pandas as pd
import neural_network as nn

# open csv
# load into a dataframe
df = pd.read_csv("column_3C_weka.csv")

#Create training input and training output
#Note ony 2/3 of data set used for trainin, 1/3 reserved for testing
training_input = df[df.out == "Hernia"].iloc[0:40, :6]
training_input = training_input.append(df[df.out == "Spondylolisthesis"].iloc[0:100, :6], ignore_index=True)
training_input = training_input.append(df[df.out == "Normal"].iloc[0:66, :6], ignore_index=True)

training_input = (training_input.transform(lambda x: x/100)).values

#Change Hernia to 1, Spondylolisthesis to 2, and Normal to 0
df.loc[df.out == "Hernia", 'out'] = 1
df.loc[df.out == "Spondylolisthesis", 'out'] = 2
df.loc[df.out == "Normal", 'out'] = 0
training_output = df.iloc[0:40, 6]
training_output = training_output.append(df.iloc[61:161, 6])
training_output = training_output.append(df.iloc[211:277, 6])
training_output = training_output.values
print(training_output)

print(training_input)
input("Press any key to continue")

# train neural network
# demonstrate predictions
SEED = 20180104
ITERATION_COUNT = 5000
DELTA = 0.02
mini_medic = nn.NeuralNetwork(6, 1, SEED, 2)
mini_medic2 = nn.NeuralNetwork(6, 1, SEED, 2)
mini_medic.train(training_input, training_output, ITERATION_COUNT)
mini_medic2.train(training_input, training_output, ITERATION_COUNT)
test_inputs = df.iloc[1,:6]
test_inputs = (test_inputs.transform(lambda x: x/100)).values
print("Initital test", test_inputs)
print(mini_medic.out_predict(test_inputs))
print(mini_medic2.out_predict(test_inputs))
input("Press any key to exit")
