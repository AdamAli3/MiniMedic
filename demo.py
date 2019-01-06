import hypothesis as H
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("column_3C_weka.csv")

h1 = H.hypothesis(["Hernia"], ["Spondylolisthesis","Normal"], [0,0,0,0,0])
h2 = H.hypothesis(["Spondylolisthesis"], ["Hernia","Normal"], [0,0,0,0,0])
h3 = H.hypothesis(["Normal"], ["Spondylolisthesis","Hernia"], [0,0,0,0,0])
h1.train(df)
h2.train(df)
h3.train(df)

x_array = np.array([72.07627839,18.94617604,50.99999999,53.13010236,114.2130126,1.01004051])
predict = [h1.predict(x_array), h2.predict(x_array), h3.predict(x_array)]
print(predict)


h1.plot_data(1);
h2.plot_data(2);
h3.plot_data(3);
