import pandas as pd
import numpy as np
import module
import proc_data

df = pd.read_csv('../../data/Heart.csv')


if __name__ == '__main__':
	module.proc(df)
	#proc_data.visualization(df)
