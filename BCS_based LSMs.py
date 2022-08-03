import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import time
import numpy as np

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
non_landslide_filepath = r'Data/Input/灾点15_18_250m缓冲区_剪切_非滑坡点.xls'
non_landslide_file = pd.read_excel(non_landslide_filepath)

# 删除空值
non_landslide_file = non_landslide_file.dropna(how='any', axis=0)

# 修改列名与值
non_landslide_file['CID'].values[:] = 0
non_landslide_file = non_landslide_file.rename(columns={'CID': 'Target'})

# 截取所需列0-17
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
n = 2  # 设定循环次数，即缓冲区采样次数
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
    positive_train = landslide_file.sample(n=positive_number, random_state=10)
    negative_train = non_landslide_sample.sample(n=positive_number, random_state=10)
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
    #  在均值和标准差旁添加坐标
    all_LSI = pd.concat([all_predict_1, all_xy], axis=1)
    # 存储为压缩格式,不要列名
    if i == 0:
        all_LSI.to_csv('./Data/output参数确认/两次采样的区别/all_LSI_250m_第3次.csv', index=False, header=False)
    elif i == 1:
        all_LSI.to_csv('./Data/output参数确认/两次采样的区别/all_LSI_250m_第4次.csv', index=False, header=False)

# 将AUC导出
all_auc.to_csv('./Data/output参数确认/两次采样的区别/all_auc_250m_3-4次.csv', index=False)

# 测试时长
time_end = time.time()
print('totally cost', time_end-time_start)

