import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import proc_data
from skfuzzy import control as ctrl

df = pd.read_csv('../../data/Heart.csv')

proc_data.visualization(df)
