import pandas as pd

# 读取mean_std文件
all_mean_std = pd.read_csv('Data/output_220624/extratree/all_mean_std_4000m_600次.csv', header=None)
all_mean_std.columns = ['mean', 'std', 'x', 'y']
all_mean = all_mean_std.drop(columns='std')

# 读取cv文件,将第一行空值删去
all_cv = pd.read_csv('Data/output_220624/extratree/all_cv_4000m_600次.csv', header=None)
all_cv.columns = ['cv']
all_cv = all_cv.drop(index=0)
all_cv = all_cv.reset_index(drop=True)

# 将mean和cv拼接在一起
all_mean_cv = pd.concat([all_mean, all_cv], axis=1)

# 改变列的顺序
order = ['mean', 'cv', 'x', 'y']
all_mean_cv = all_mean_cv[order]

# cv的断点值是固定的
cv_breaks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 2]

# 以下为RF_1000_400的断点值
# mean_breaks = [0, 0.2012, 0.346739, 0.502775, 0.674034, 0.9823]
# var_breaks = [0, 0.006411, 0.010402, 0.013949, 0.01794, 0.024591, 0.113272]
# 以下为RF_500_400的断点值
# mean_breaks = [0, 0.208433999, 0.34867025, 0.500277008, 0.663254273, 0.981628466]
# var_breaks = [0, 0.006383741, 0.010334956, 0.013847146, 0.01779836, 0.02350567, 0.112188479]
# 以下为RF_250_400的断点值
# mean_breaks = [0, 0.208750249, 0.348720413, 0.496256532, 0.655141582, 0.980477639]
# var_breaks = [0, 0.006781675, 0.010956251, 0.014666986, 0.018841563, 0.025335348, 0.118567558]
# 以下为RF_2000_400的断点值
# mean_breaks = [0, 0.199516412, 0.356604811, 0.525187484, 0.701433005, 0.988789833]
# var_breaks = [0, 0.006067881, 0.009963997, 0.013373099, 0.017269215, 0.023113389, 0.124412404]
# 以下为RF_100_400的断点值 （被采用为固定值不变）
# mean_breaks = [0, 0.209715674, 0.346086213, 0.493820963, 0.649131854, 1]
# var_breaks = [0, 0.006859962, 0.010796013, 0.014294724, 0.018230775, 0.02435352, 0.11182131]

mean_breaks = [0, 0.214723, 0.350205, 0.496978, 0.658804, 1]

# 创建一个空的c存储最终值
all_mean_cv['c'] = 0

def function(a, b):
    for i in range(5):
        for j in range(6):
            if (mean_breaks[i] < a <= mean_breaks[i + 1]) and (cv_breaks[j] < b <= cv_breaks[j + 1]):
                return (i+1)*10+j+1
all_mean_cv['c'] = all_mean_cv.apply(lambda x: function(x['mean'], x['cv']), axis=1)

all_mean_cv.iloc[:, 2:].to_csv('Data/output_最终LSCM/extratree/all_meancv_index_600次_4000m.csv', index=False, header=False)

