import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import time
import numpy as np

random.seed(10)
time_start = time.time()


def base_model(model_type=1):
    if model_type == 1:
        model = RandomForestClassifier(max_depth=17.049876353965335, max_features=0.2129133497401154,
                                       min_samples_split=2, n_estimators=739)
    elif model_type == 2:
        model = ExtraTreesClassifier(max_depth=15, max_features=0.24799180879423444, min_samples_split=2,
                                     n_estimators=400)
    return model


#  1.读取非滑坡数据进行处理
non_landslide_filepath = r'Data/Input/灾点15_18_4000m缓冲区_剪切_非滑坡点.xls'
non_landslide_file = pd.read_excel(non_landslide_filepath)
print('开始RF_4000m的计算')


# 删除含有空值的滑坡点
non_landslide_file = non_landslide_file.dropna(how='any', axis=0)

# 修改列名与值
non_landslide_file['CID'].values[:] = 0
non_landslide_file = non_landslide_file.rename(columns={'CID': 'Target'})

# 截取所需列0-17（经纬坐标及15个因子）
non_landslide_file = non_landslide_file.iloc[:, 0:17]

#  2.读取滑坡数据,统一列名
landslide_file = pd.read_csv('./Data/point15年positive.csv')
landslide_file.columns = non_landslide_file.columns.values

#  读取所有点位的数据
all_data_path = 'Data/安化县15年所有点含经纬坐标.csv'
all_data = pd.read_csv(all_data_path)

#  减小all_data的数量进行试算
#  all_data = all_data.iloc[:100, :]

# 分出15个因子与相应坐标
all_15 = all_data.iloc[:, 1:16]
all_xy = all_data.iloc[:, 16:]

#  开始循环
n = 600  # 设定循环次数，即缓冲区采样次数
train_ratio = 0.7  # 设定训练比率
sample_number = 492
positive_number = round(sample_number*train_ratio)
negative_number = round(sample_number*(1-train_ratio))

#  创建AUC信息的数据框
df_columns = ['auc_test', 'tn', 'fp', 'fn', 'tp', 'sensitivity', 'specificity', 'accuracy']
all_auc = pd.DataFrame(columns=df_columns)
#  创建所有点位易发性指数的数据框
all_LSI = all_data.iloc[:, 0]

# 这里如果扩充数据集的话，考虑改为采用不放回采样
for i in range(n):
    print('开始第'+str(i)+'次采样')
    non_landslide_sample = non_landslide_file.sample(n=sample_number)  # 采1:1的负样本
    # 4.进行抽样.切割训练集与测试集
    positive_train = landslide_file.sample(n=positive_number)
    negative_train = non_landslide_sample.sample(n=positive_number)
    positive_test = landslide_file[~landslide_file.index.isin(positive_train.index)]
    negative_test = non_landslide_sample[~non_landslide_sample.index.isin(negative_train.index)]
    train = pd.concat([positive_train, negative_train])
    test = pd.concat([positive_test, negative_test])
    x_train = train.iloc[:, 2:]
    x_test = test.iloc[:, 2:]
    y_train = train.iloc[:, 1:2]
    y_train = np.ravel(y_train)
    y_test = test.iloc[:, 1:2]
# 创建模型
    clf = base_model()
    clf.fit(x_train, y_train)
    x_predict = clf.predict_proba(x_test)
    auc_test = roc_auc_score(y_test, x_predict[:, 1])
# 混淆矩阵
    y_pred = clf.predict(x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# 统计学指标
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    auc = pd.DataFrame([[auc_test, tn, fp, fn, tp, sensitivity, specificity, accuracy]], columns=df_columns)
# 计算所有次数的不同指标，并存储在循环外
    all_auc = pd.concat([all_auc, auc], ignore_index=True)
# 计算所有点位的易发性指数，并存储在循环外
    all_predict = clf.predict_proba(all_15)
    all_predict_1 = pd.DataFrame(all_predict[:, 1])
    all_LSI = pd.concat([all_LSI, all_predict_1], axis=1, ignore_index=True)
#  由于内存不够，每计算一百次，计算其均值与标准差，最后计算总的均值与标准差并存储起来
    if i == 99:
        all_mean_part1 = all_LSI.iloc[:, 1:].mean(axis=1)
        all_std_part1 = all_LSI.iloc[:, 1:].std(axis=1)
        all_mean_std_part1 = pd.concat([all_mean_part1, all_std_part1], axis=1)
        all_mean_std_part1 = all_mean_std_part1.rename(columns={0: 'mean', 1: 'std'})
        all_LSI = all_data.iloc[:, 0]  # 重新制作一个数据框放置易发性指数
        print("采样次数达到100次，计算均值和标准差")
    if i == 199 or i == 299 or i == 399 or i == 499 or i == 599:
        all_mean_part2 = all_LSI.iloc[:, 1:].mean(axis=1)
        all_std_part2 = all_LSI.iloc[:, 1:].std(axis=1)
        all_mean_std_part2 = pd.concat([all_mean_part2, all_std_part2], axis=1)
        all_mean_std_part2 = all_mean_std_part2.rename(columns={0: 'mean', 1: 'std'})
        #  合并所有mean和std
        if i == 199:
            all_mean = (all_mean_std_part1['mean'] + all_mean_std_part2['mean']) / 2
            all_std = 100 * (all_mean_std_part1['std'] ** 2 + all_mean_std_part2['std'] ** 2) + (10000/200) * (
                        all_mean_std_part2['mean'] - all_mean_std_part1['mean']) ** 2
            all_std = (all_std/200)**0.5
            print("采样次数达到200次，计算均值和标准差")
        elif i == 299:
            all_mean = (200*all_mean_std_part1['mean'] + 100*all_mean_std_part2['mean']) / 300
            all_std = (200*all_mean_std_part1['std'] ** 2 + 100*all_mean_std_part2['std'] ** 2) + (20000/300) * (
                        all_mean_std_part2['mean'] - all_mean_std_part1['mean']) ** 2
            all_std = (all_std / 300) ** 0.5
            print("采样次数达到300次，计算均值和标准差")
        elif i == 399:
            all_mean = (300*all_mean_std_part1['mean'] + 100*all_mean_std_part2['mean']) / 400
            all_std = (300*all_mean_std_part1['std'] ** 2 + 100*all_mean_std_part2['std'] ** 2) + (30000/400) * (
                        all_mean_std_part2['mean'] - all_mean_std_part1['mean']) ** 2
            all_std = (all_std/400)**0.5
            print("采样次数达到400次，计算均值和标准差")
        elif i == 499:
            all_mean = (400 * all_mean_std_part1['mean'] + 100 * all_mean_std_part2['mean']) / 500
            all_std = (400 * all_mean_std_part1['std'] ** 2 + 100 * all_mean_std_part2['std'] ** 2) + (40000 / 500) * (
                    all_mean_std_part2['mean'] - all_mean_std_part1['mean']) ** 2
            all_std = (all_std / 500) ** 0.5
            print("采样次数达到500次，计算均值和标准差")
        elif i == 599:
            all_mean = (500 * all_mean_std_part1['mean'] + 100 * all_mean_std_part2['mean']) / 600
            all_std = (500 * all_mean_std_part1['std'] ** 2 + 100 * all_mean_std_part2['std'] ** 2) + (50000 / 600) * (
                    all_mean_std_part2['mean'] - all_mean_std_part1['mean']) ** 2
            all_std = (all_std / 600) ** 0.5
            print("采样次数达到600次，计算均值和标准差")


        all_mean_std_part1 = pd.concat([all_mean, all_std], axis=1)
        all_mean_std_part1.columns = ['mean', 'std']
        all_LSI = all_data.iloc[:, 0]  # 重新制作一个数据框放置易发性指数

# 将AUC导出
all_auc.to_csv('./Data/output_220624/RF/all_auc_4000m_600次.csv', index=False)

# 基于均值和标准差计算变异系数cv
all_cv = all_std/all_mean
all_cv.to_csv('./Data/output_220624/RF/all_cv_4000m_600次.csv', index=False)

#  在均值和标准差旁添加坐标
all_mean_std = pd.concat([all_mean_std_part1, all_xy], axis=1)

# 存储为压缩格式,不要列名
all_mean_std.to_csv('./Data/output_220624/RF/all_mean_std_4000m_600次.csv', index=False, header=False)
# all_mean_var_test = pd.read_csv('./Data/output/all_mean_var_100m_100次.gz')

# 测试时长
time_end = time.time()
print('totally cost', time_end-time_start)

