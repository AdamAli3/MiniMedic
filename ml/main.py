import numpy as np
import pandas as pd

# open csv
# load into a dataframe
df = pd.read_csv("column_3C_weka.csv")

#Create testing input and testing output
#Note ony 2/3 of data set used for trainin, 1/3 reserved for testing
testing_input = df[df.out == "Hernia"].iloc[0:40, :6]
testing_input = testing_input.append(df[df.out == "Spondylolisthesis"].iloc[0:100, :6], ignore_index=True)
testing_input = testing_input.append(df[df.out == "Normal"].iloc[0:66, :6], ignore_index=True)

#Change Hernia to 1, Spondylolisthesis to 0.5, and Normal to 0
df.loc[df.out == "Hernia", 'out'] = 1
df.loc[df.out == "Spondylolisthesis", 'out'] = 0.5
df.loc[df.out == "Normal", 'out'] = 0

testing_output = df.iloc[0:40, 6]
testing_output = testing_output.append(df.iloc[61:161, 6])
testing_output = testing_output.append(df.iloc[211:277, 6])

testing_output = testing_output.values

input("Press any key to continue")
