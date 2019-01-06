import hypothesis as H
import numpy as np
import pandas as pd

df = pd.read_csv("column_3C_weka.csv")

h1 = H.hypothesis(["Hernia"], ["Spondylolisthesis","Normal"], [0,0,0,0,0])
h1.train(df)
print(h1.theta)

prediction = h1.predict(np.array([39.05695098,10.06099147,25.01537822,28.99595951,114.4054254,4.564258645]))
print(prediction)
