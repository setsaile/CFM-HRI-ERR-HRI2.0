import os
import pandas as pd
import numpy as np
from datetime import timedelta
from glob import glob
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix
)
import numpy as np
from sklearn.metrics import roc_auc_score
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

train_data_path = "./BASE/Data/rf_train_window_size_3_stride_0.5.csv"
val_data_path = "./BASE/Data/rf_val_window_size_3_stride_0.5.csv"
test_data_path = "./BASE/Data/rf_test_window_size_3_stride_0.5.csv"

train_data = pd.read_csv(train_data_path, low_memory=False)
val_data = pd.read_csv(val_data_path, low_memory=False)
test_data = pd.read_csv(test_data_path, low_memory=False)


n_windows = len(train_data)
print(f"Number of windows: {n_windows}")

n_features = len(train_data.columns) 
print(f"Number of features: {n_features}")

positive_rate = (train_data['robot_error'] == 1).sum() / len(train_data['robot_error']) * 100
print(f"Positive Rate: {positive_rate}%")

# 这进行了修改,test->train
LABEL_FILE_PATTERNS = {
    "robot_errors": "challenge1_robot_error_labels_{task}_train.csv",
    "human_reactions_ch1": "challenge1_user_reaction_labels_{task}_train.csv",
    "human_reactions_ch2": "challenge2_user_reaction_labels_{task}_train.csv"
}

def fix_timestamp_format(ts):
    if isinstance(ts, str):
        parts = ts.split(':')
        if len(parts) == 4:
            return ':'.join(parts[:3]) + '.' + parts[3]
    return ts

"""Load all label files for a specific task and preprocess timestamps"""
# Base Path to Data
# 进行修改
BASE_PATH = "./data_BASE"

def load_and_preprocess_labels(task):
    """Load all label files for a specific task and preprocess timestamps"""
    labels = {}    
    
    try:
        # Load robot error labels
        robot_errors = pd.read_csv(
            f"{BASE_PATH}/labels_train/challenge1_train/{LABEL_FILE_PATTERNS['robot_errors'].format(task=task)}"
        )
        robot_errors['error_onset'] = robot_errors['error_onset'].apply(fix_timestamp_format)
        robot_errors['error_onset'] = pd.to_timedelta(robot_errors['error_onset'])
        robot_errors['error_offset'] = robot_errors['error_offset'].apply(fix_timestamp_format)
        robot_errors['error_offset'] = pd.to_timedelta(robot_errors['error_offset'])
        robot_errors['trial'] = robot_errors['trial_name'].apply(lambda s: s.split('-', 1)[0])
        labels['robot_errors'] = robot_errors        
        
        # Load human reaction labels
        human_reactions_ch1 = pd.read_csv(
            f"{BASE_PATH}/labels_train/challenge1_train/{LABEL_FILE_PATTERNS['human_reactions_ch1'].format(task=task)}"
        )
        human_reactions_ch1['reaction_onset'] = human_reactions_ch1['reaction_onset'].apply(fix_timestamp_format)
        human_reactions_ch1['reaction_onset'] = pd.to_timedelta(human_reactions_ch1['reaction_onset'])
        human_reactions_ch1['reaction_offset'] = human_reactions_ch1['reaction_offset'].apply(fix_timestamp_format)
        human_reactions_ch1['reaction_offset'] = pd.to_timedelta(human_reactions_ch1['reaction_offset'])
        human_reactions_ch1['trial'] = human_reactions_ch1['trial_name'].apply(lambda s: s.split('-', 1)[0])
        labels['human_reactions_ch1'] = human_reactions_ch1        
        
        human_reactions_ch2 = pd.read_csv(
            f"{BASE_PATH}/labels_train/challenge2_train/{LABEL_FILE_PATTERNS['human_reactions_ch2'].format(task=task)}"
        )
        human_reactions_ch2['reaction_onset'] = human_reactions_ch2['reaction_onset'].apply(fix_timestamp_format)
        human_reactions_ch2['reaction_onset'] = pd.to_timedelta(human_reactions_ch2['reaction_onset'])
        human_reactions_ch2['reaction_offset'] = human_reactions_ch2['reaction_offset'].apply(fix_timestamp_format)
        human_reactions_ch2['reaction_offset'] = pd.to_timedelta(human_reactions_ch2['reaction_offset'])
        human_reactions_ch2['trial'] = human_reactions_ch2['trial_name'].apply(lambda s: s.split('-', 1)[0])
        labels['human_reactions_ch2'] = human_reactions_ch2    
    
    except Exception as e:
        print(f"Error loading label files for task {task}: {str(e)}")
        return None    
    
    return labels, robot_errors, human_reactions_ch1, human_reactions_ch2

all_robot_errors = pd.DataFrame()
all_human_reactions_ch1 = pd.DataFrame()
all_human_reactions_ch2 = pd.DataFrame()

tasks = train_data['task'].unique()
print("task:",tasks)
for task in tasks:
    print(f"\n  Task: {task}")            
    
    # Load label files for this task
    labels, robot_errors, human_reactions_ch1, human_reactions_ch2 = load_and_preprocess_labels(task)
    robot_errors['task'] = task
    human_reactions_ch1['task'] = task
    human_reactions_ch2['task'] = task
    all_robot_errors = pd.concat([all_robot_errors, robot_errors], ignore_index=True)
    all_human_reactions_ch1 = pd.concat([all_human_reactions_ch1, human_reactions_ch1], ignore_index=True)
    all_human_reactions_ch2 = pd.concat([all_human_reactions_ch2, human_reactions_ch2], ignore_index=True)
    
    
def evaluate_reaction(y_pred_df, y_true_df):
    """
    Compute classification metrics and count of detected error events.
    
    Parameters
    ----------
    y_pred_df : pd.DataFrame
        DataFrame must contain 'task', 'trial', 'start', 'end', 'y_pred_reaction',
    y_pred_df : pd.DataFrame
        DataFrame must contain 'task', 'trial', 'reaction_onset', 'reaction_offset'.
        
    Returns
    -------
    tp: number of true positives 
    fp: number of false positives
    total_reactions: number of reactions in total
    """

    # Make sure y_true only include trials in pred
    evaluation_trials = y_pred_df['trial'].astype(str).unique()
    y_true_df = y_true_df.loc[y_true_df['trial'].isin(evaluation_trials)]
    
    # Look for true positive and false positive 
    pos_pred = y_pred_df.loc[y_pred_df['y_pred_reaction'] == 1]
    pos_pred['id'] = pos_pred.index
    pos_pred['overlap_reaction'] = 0

    # initialize metrics
    tp = 0
    fp = 0
    total_reactions = len(y_true_df)

    # check that the trials contain reactions 
    if len(y_true_df) > 0:
        for _, row in y_true_df.iterrows():
            task = row['task']
            trial = row['trial']
            reaction_onset = row['reaction_onset'].total_seconds() - 1 # added one second tolerance
            reaction_offset = row['reaction_offset'].total_seconds() + 1 # added one second tolerance

            # check if the predicted reaction overlaps with actual reaction 
            detected_err = pos_pred[(pos_pred['task'] == task) & (pos_pred['trial'] == trial) & 
                                    ((pos_pred['start'] >= reaction_onset) & (pos_pred['start'] <= reaction_offset)) |
                                    ((pos_pred['end']   >= reaction_onset) & (pos_pred['end']   <= reaction_offset)) |
                                    ((pos_pred['start'] <= reaction_onset) & (pos_pred['end'] >= reaction_offset))]
            pos_pred.loc[pos_pred['id'].isin(detected_err['id']), 'overlap_reaction'] = 1
            if len(detected_err) > 0: 
                tp = tp + 1

        # prediction is a false positive if it does not overlap with actual reaction
        fp = len(pos_pred.loc[pos_pred['overlap_reaction'] == 0])
        
        print(f"True Positive: {tp} ({tp / total_reactions * 100}/%)")
        print(f"False Positive: {fp}")

        fn = total_reactions - tp
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) 
        print(f"F1: {f1}")

    return tp, fp, total_reactions

from imblearn.over_sampling import SMOTE    # SMOTE是用来插值采样的,解决label不平衡的方法
from sklearn.impute import SimpleImputer    # 用于填补NaN的工具类
import joblib # 保存sklearn模型

# split
df_train = train_data.copy()
df_val = val_data.copy()
df_test = test_data.copy()

# 丢弃非特征列
X_train = df_train.drop(columns=[
    'start','end','robot_error','reaction_ch1','reaction_ch2',
    'reaction_type','system','task','trial'
])
X_val = df_val.drop(columns=[
    'start','end','robot_error','reaction_ch1','reaction_ch2',
    'reaction_type','system','task','trial'
])
X_test = df_test.drop(columns=[
    'start','end','system','task','trial'
])

y_reaction_train = df_train['reaction_ch2']
y_reaction_val   = df_val['reaction_ch2'] 

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)


# with SMOTE
smote = SMOTE(random_state=42)  # 这个也是设置随机种子,保证每一次生成一样的值
X_train_sm, y_train_sm = smote.fit_resample(X_train_imputed, y_reaction_train)
X_train_sm = pd.DataFrame(X_train_sm, columns=X_train.columns)
# 将2D -> 3D升维,变成相应的形式

# 将 Pandas DataFrame 转为 NumPy 并 reshape 成 MiniRocket 需要的格式
X_train_sm = X_train_sm.to_numpy().reshape((X_train_sm.shape[0], X_train_sm.shape[1], 1))
X_train_sm = np.transpose(X_train_sm, (0, 2, 1))
# 保证标签是 1D 向量
y_train_sm = y_train_sm.to_numpy().ravel()

print("X_train_sm.shape:", X_train_sm.shape)
print("y_train_sm.shape:", y_train_sm.shape)

from tqdm import tqdm
import numpy as np
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier
from sklearn.pipeline import make_pipeline

class MiniRocketWithProgress(MiniRocket):
    """带进度条的MiniRocket"""
    
    def fit(self, X, y=None):
        print(f"   Fitting MiniRocket with {self.num_kernels} kernels...")
        
        # 模拟进度（因为无法直接访问内部循环）
        with tqdm(total=100, desc="MiniRocket Training", unit="%") as pbar:
            # 调用原始的fit方法
            import threading
            import time
            
            # 启动训练线程
            result = [None]
            exception = [None]
            
            def train():
                try:
                    result[0] = super(MiniRocketWithProgress, self).fit(X, y)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=train)
            thread.start()
            
            # 模拟进度更新
            progress = 0
            while thread.is_alive():
                if progress < 90:  # 保留最后10%给完成
                    progress += np.random.randint(1, 5)
                    pbar.n = min(progress, 90)
                    pbar.refresh()
                time.sleep(2)  # 每2秒更新一次
            
            thread.join()
            
            if exception[0]:
                raise exception[0]
            
            pbar.n = 100
            pbar.refresh()
            
        return result[0] if result[0] else self


model_path = "./BASE/models/clf_ch2_ridgeclass_win3_stri0.5.pkl"

if os.path.exists(model_path):
    print("Loading model from disk...")
    clf_reaction = joblib.load(model_path)
else:
    print(" • Training challenge 2 model with MiniRocket and RidgeClassifierCV…")
    
    # 使用更多内核以提高特征提取能力
    minirocket = MiniRocketWithProgress(
        num_kernels=2000,  # 从5000降低到2000，减少特征维度
        random_state=42
    )
    
    # 使用RidgeClassifierCV替代LogisticRegressionCV
    # RidgeClassifierCV在高维特征空间中通常表现更好
    classifier = RidgeClassifierCV(
        alphas=np.logspace(-1, 2, 8),  # 缩小正则化参数范围，避免过小的alpha值
        cv=5,
        class_weight='balanced',  # 处理类别不平衡
        scoring='f1',  # 使用F1作为评分标准，更关注TP
        # normalize=True,  # 添加特征标准化
        # solver='auto'  # 让sklearn自动选择最稳定的求解器
    )
    
    # 分步进行以显示更详细的进度
    print("   Step 1: Fitting MiniRocket transformer...")
    minirocket.fit(X_train_sm)
    
    print("   Step 2: Extracting features...")
    with tqdm(desc="Feature extraction", unit="samples") as pbar:
        X_transformed = minirocket.transform(X_train_sm)
        pbar.update(X_train_sm.shape[0])
    
    print("   Step 3: Training classifier...")
    with tqdm(desc="Classifier training") as pbar:
        classifier.fit(X_transformed, y_train_sm)
        pbar.update(1)
    
    print("   Step 4: Creating pipeline and saving...")
    clf_reaction = make_pipeline(minirocket, classifier)
    joblib.dump(clf_reaction, model_path)
    
    print(" done (MiniRocket model trained and saved).")

# predict
print("模型类型：", type(clf_reaction))
X_val = X_val.to_numpy().reshape((X_val.shape[0], X_val.shape[1], 1))
X_val = np.transpose(X_val, (0, 2, 1))

# RidgeClassifier没有predict_proba方法，使用decision_function获取决策分数
scores_reaction = clf_reaction.decision_function(X_val)

# 归一化决策分数到0-1之间，便于设置阈值
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
probs_reaction = scaler.fit_transform(scores_reaction.reshape(-1, 1)).flatten()

# 使用更高的阈值以减少FP
threshold = 0.8  # 提高阈值以减少FP
pred_reaction = (probs_reaction >= threshold).astype(int)

# 对测试集做同样处理
X_test = X_test.to_numpy().reshape((X_test.shape[0], X_test.shape[1], 1))
X_test = np.transpose(X_test, (0, 2, 1))
scores_reaction_test = clf_reaction.decision_function(X_test)
probs_reaction_test = scaler.transform(scores_reaction_test.reshape(-1, 1)).flatten()
pred_reaction_test = (probs_reaction_test >= threshold).astype(int)

# compute reaction-model metrics
f1_e = f1_score(y_reaction_val, pred_reaction, average='macro')
acc_e  = accuracy_score(y_reaction_val, pred_reaction)
if sum(y_reaction_val) > 0: 
    tn, fp, fn, tp = confusion_matrix(y_reaction_val, pred_reaction).ravel()
else: 
    tn, fp, fn, tp = 0, 0, 0, 0

tpr_e = tp / (tp + fn) 
fpr_e = fp / (fp + tn) 
tnr_e = tn / (tn + fp) 
fnr_e = fn / (fn + tp) 
auc_e = roc_auc_score(y_reaction_val, probs_reaction) 

print(
    f"AUC: {auc_e:.3f}, " 
    f"F1: {f1_e:.3f}, Acc: {acc_e:.3f}, "
    f"TPR: {tpr_e:.3f}, FPR: {fpr_e:.3f}, "
    f"TNR: {tnr_e:.3f}, FNR: {fnr_e:.3f}, "
)

df_val['y_pred_reaction'] = pred_reaction
df_test['y_pred_reaction'] = pred_reaction_test
val_trials = df_val['trial'].astype(str).unique()

# 保存test形式
pred_test_df = df_test[['task', 'trial', 'start', 'end', 'y_pred_reaction']]
pred_test_df.to_csv("./BASE/result/y_pred_ch2_w3_s0.5_Ridge.csv", index = False)


# # 下面是化为化为标准形式
# # 化成标准形式
# # 只保留预测为 1 的记录
# pred_test_df = pred_test_df[pred_test_df['y_pred_reaction'] == 1].reset_index(drop=True)

# # 如果为空直接返回空结果
# if pred_test_df.empty:
#     merged_df = pd.DataFrame(columns=['task', 'trial', 'start', 'end', 'y_pred_reaction'])
# else:
#     # 时间格式转换函数
#     def format_time(seconds):
#         td = timedelta(seconds=seconds)
#         total_seconds = int(td.total_seconds())
#         milliseconds = int((td.total_seconds() - total_seconds) * 1000)
#         hours = total_seconds // 3600
#         minutes = (total_seconds % 3600) // 60
#         seconds = total_seconds % 60
#         return f"{hours:02}:{minutes:02}:{seconds:02}:{milliseconds:03}"

#     # 自动推测滑动步长
#     if len(pred_test_df) >= 2:
#         step_size = round(pred_test_df['start'].iloc[1] - pred_test_df['start'].iloc[0], 3)
#     else:
#         step_size = 0.5  # 默认值

#     # 按照 task、trial 排序，确保合并顺序正确
#     pred_test_df = pred_test_df.sort_values(by=['task', 'trial', 'start']).reset_index(drop=True)

#     # 合并逻辑
#     merged = []
#     prev = pred_test_df.iloc[0].copy()

#     for idx in range(1, len(pred_test_df)):
#         curr = pred_test_df.iloc[idx]

#         is_same_group = (
#             curr['task'] == prev['task'] and
#             curr['trial'] == prev['trial'] and
#             curr['y_pred_reaction'] == prev['y_pred_reaction'] and
#             (curr['start'] - prev['end'] <= step_size + 1e-6)
#         )

#         if is_same_group:
#             # 合并时间段
#             prev['end'] = max(prev['end'], curr['end'])
#         else:
#             merged.append(prev.copy())
#             prev = curr.copy()

#     merged.append(prev.copy())  # 最后一段别忘了加

#     # 转为 DataFrame
#     merged_df = pd.DataFrame(merged)

#     # 时间格式化
#     merged_df['start'] = merged_df['start'].apply(format_time)
#     merged_df['end'] = merged_df['end'].apply(format_time)

# # 最终列输出顺序
# merged_df = merged_df[['task', 'trial', 'start', 'end', 'y_pred_reaction']]

# # 上面是化为标准形式



# merged_df.to_csv("./BASE/result/y_pred_ch2_first.csv", index = False)
# val_trials = df_val['trial'].astype(str).unique()

# 分别取预测标签和真实标签
y_pred_df = df_val[['task', 'trial', 'start', 'end', 'y_pred_reaction']]
# # 使用下面的作为y_true会引入label为0的行,与评估逻辑不同,会导致错误的fp和tp
# y_true_df = df_val[['task', 'trial', 'start', 'end', 'reaction_ch2']]
y_true_df = all_human_reactions_ch2[['task', 'trial', 'reaction_onset', 'reaction_offset']].loc[all_human_reactions_ch2['trial'].isin(val_trials)]


# 查看 y_pred_df 中 trial 列的所有唯一类别名称
print("y_pred_df trial 列唯一类别名称：")
print(y_pred_df['trial'].unique())

# 查看 y_true_df 中 trial 列的所有唯一类别名称
print("y_true_df trial 列唯一类别名称：")
print(y_true_df['trial'].unique())

tp, fp, total_reaction = evaluate_reaction(y_pred_df, y_true_df)

