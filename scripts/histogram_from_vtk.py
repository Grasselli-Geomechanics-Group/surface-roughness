import os, glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
sns.set_context("paper",font_scale=2)
folder = r'X:\20220113_surface_sampling_samples'
file_prefix = '20220823_beza_direct_tension_bot_1_1.stl_delta_t_r2.5'

x = 'max_unidirectional_magnitude-delta_t'
data = pd.DataFrame()
files = glob.glob(folder+"\*.stl")
for file in files:
    magnitude_file = os.path.join(folder,file + "_delta_t_r2.5.csv")


    df = pd.read_csv(magnitude_file)
    df['file'] = os.path.basename(file)
    data = pd.concat([data,df],axis=0)

data = data.rename(columns={"max_unidirectional_magnitude-delta_t":"Maximum Unidirectional Delta T"})
sns.kdeplot(data,x='Maximum Unidirectional Delta T',hue='file')

plt.savefig(os.path.join(folder,f'maximum_unidirectional_plots.svg'))