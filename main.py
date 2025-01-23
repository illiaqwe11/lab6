import pandas as pd
from medical_data_visualizer import *

df = pd.read_csv('medical_examination.csv')

df = preprocess_data(df)

cat_fig = draw_cat_plot(df)
cat_fig.savefig('cat_plot.png') 

heat_fig = draw_heat_map(df)
heat_fig.savefig('heat_map.png')  
