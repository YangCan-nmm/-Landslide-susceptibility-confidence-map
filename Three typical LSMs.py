import pandas as pd
import numpy as np
all_mean_std = pd.read_csv('Data/output_220624/extratree/all_mean_std_100m_600次.csv', header=None)
all_mean_std.columns = ['mean', 'std', 'x', 'y']

all_mean_std['upper'] = all_mean_std['mean']+3*all_mean_std['std']
all_mean_std['upper'][all_mean_std.upper > 1] = 1

all_mean_std['lower'] = all_mean_std['mean']-3*all_mean_std['std']
all_mean_std['lower'][all_mean_std.lower < 0] = 0

all_mean_std.to_csv('Data/output_上下限/extra/all_mean_std_xy_upper_lower_100m.csv', index=False, header=False)
