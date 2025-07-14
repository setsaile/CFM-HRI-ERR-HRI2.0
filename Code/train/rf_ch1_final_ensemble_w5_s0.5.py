import os
import pandas as pd
import numpy as np
from datetime import timedelta
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
import gc  # 添加垃圾回收模块
from scipy.special import expit  # sigmoid function
# 导入必要的库
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, 
    roc_auc_score, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.impute import KNNImputer
from sktime.transformations.panel.rocket import MiniRocket
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin

# 数据路径设置
train_data_paths = {
    "w5_s0.5": "./BASE/Data/rf_train_window_size_5_stride_0.5.csv",
    "w3_s0.5": "./BASE/Data/rf_train_window_size_3_stride_0.5.csv",
    "w1_s1": "./BASE/Data/rf_train_window_size_1_stride_1.csv",
    "w10_s4": "./BASE/Data/rf_train_window_size_10_stride_4.csv"
}

val_data_paths = {
    "w5_s0.5": "./BASE/Data/rf_val_window_size_5_stride_0.5.csv",
    "w3_s0.5": "./BASE/Data/rf_val_window_size_3_stride_0.5.csv",
    "w1_s1": "./BASE/Data/rf_val_window_size_1_stride_1.csv",
    "w10_s4": "./BASE/Data/rf_val_window_size_10_stride_4.csv"
}

test_data_paths = {
    "w5_s0.5": "./BASE/Data/rf_test_window_size_5_stride_0.5.csv",
    "w3_s0.5": "./BASE/Data/rf_test_window_size_3_stride_0.5.csv",
    "w1_s1": "./BASE/Data/rf_test_window_size_1_stride_1.csv",
    "w10_s4": "./BASE/Data/rf_test_window_size_10_stride_4.csv"
}

# 主要使用的数据集
main_config = "w5_s0.5"
train_data_path = train_data_paths[main_config]
val_data_path = val_data_paths[main_config]
test_data_path = test_data_paths[main_config]

train_data = pd.read_csv(train_data_path, low_memory=False)
val_data = pd.read_csv(val_data_path, low_memory=False)
test_data = pd.read_csv(test_data_path, low_memory=False)

n_windows = len(train_data)
print(f"Number of windows: {n_windows}")

n_features = len(train_data.columns) 
print(f"Number of features: {n_features}")

positive_rate = (train_data['robot_error'] == 1).sum() / len(train_data['robot_error']) * 100
print(f"Positive Rate: {positive_rate}%")

# 标签文件模式
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

# 基础数据路径
BASE_PATH = "./data_BASE"

def load_and_preprocess_labels(task):
    """加载并预处理特定任务的所有标签文件"""
    labels = {}    
    
    try:
        # 加载机器人错误标签
        robot_errors = pd.read_csv(
            f"{BASE_PATH}/labels_train/challenge1_train/{LABEL_FILE_PATTERNS['robot_errors'].format(task=task)}"
        )
        robot_errors['error_onset'] = robot_errors['error_onset'].apply(fix_timestamp_format)
        robot_errors['error_onset'] = pd.to_timedelta(robot_errors['error_onset'])
        robot_errors['error_offset'] = robot_errors['error_offset'].apply(fix_timestamp_format)
        robot_errors['error_offset'] = pd.to_timedelta(robot_errors['error_offset'])
        robot_errors['trial'] = robot_errors['trial_name'].apply(lambda s: s.split('-', 1)[0])
        labels['robot_errors'] = robot_errors        
        
        # 加载人类反应标签 - 挑战1
        human_reactions_ch1 = pd.read_csv(
            f"{BASE_PATH}/labels_train/challenge1_train/{LABEL_FILE_PATTERNS['human_reactions_ch1'].format(task=task)}"
        )
        human_reactions_ch1['reaction_onset'] = human_reactions_ch1['reaction_onset'].apply(fix_timestamp_format)
        human_reactions_ch1['reaction_onset'] = pd.to_timedelta(human_reactions_ch1['reaction_onset'])
        human_reactions_ch1['reaction_offset'] = human_reactions_ch1['reaction_offset'].apply(fix_timestamp_format)
        human_reactions_ch1['reaction_offset'] = pd.to_timedelta(human_reactions_ch1['reaction_offset'])
        human_reactions_ch1['trial'] = human_reactions_ch1['trial_name'].apply(lambda s: s.split('-', 1)[0])
        labels['human_reactions_ch1'] = human_reactions_ch1        
        
        # 加载人类反应标签 - 挑战2
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

# 初始化空的DataFrame用于存储所有标签
all_robot_errors = pd.DataFrame()
all_human_reactions_ch1 = pd.DataFrame()
all_human_reactions_ch2 = pd.DataFrame()

# 遍历所有任务加载标签
tasks = train_data['task'].unique()
print("task:", tasks)
for task in tasks:
    print(f"\n  Task: {task}")            
    
    # 加载此任务的标签文件
    try:
        labels, robot_errors, human_reactions_ch1, human_reactions_ch2 = load_and_preprocess_labels(task)
        if labels is not None:
            robot_errors['task'] = task
            human_reactions_ch1['task'] = task
            human_reactions_ch2['task'] = task
            all_robot_errors = pd.concat([all_robot_errors, robot_errors], ignore_index=True)
            all_human_reactions_ch1 = pd.concat([all_human_reactions_ch1, human_reactions_ch1], ignore_index=True)
            all_human_reactions_ch2 = pd.concat([all_human_reactions_ch2, human_reactions_ch2], ignore_index=True)
    except Exception as e:
        print(f"Error processing task {task}: {str(e)}")
        continue
    
    
def evaluate_error(y_pred_df, y_true_df):
    """
    计算分类指标和检测到的错误事件数量。
    
    参数
    ----------
    y_pred_df : pd.DataFrame
        DataFrame必须包含'task', 'trial', 'start', 'end', 'y_pred_err',
    y_true_df : pd.DataFrame
        DataFrame必须包含'task', 'trial', 'error_onset', 'error_offset'.
        
    返回
    -------
    tp: 真阳性数量
    fp: 假阳性数量
    total_errors: 总错误数量
    """

    # 确保y_true只包含预测中的试验
    evaluation_trials = y_pred_df['trial'].astype(str).unique()
    y_true_df = y_true_df.loc[y_true_df['trial'].isin(evaluation_trials)]
    
    # 寻找真阳性和假阳性
    pos_pred = y_pred_df.loc[y_pred_df['y_pred_err'] == 1].copy()  # 使用.copy()避免SettingWithCopyWarning
    pos_pred['id'] = pos_pred.index
    pos_pred['overlap_error'] = 0

    # 初始化指标
    tp = 0
    fp = 0
    total_errors = len(y_true_df)

    # 检查试验是否包含错误
    if len(y_true_df) > 0:
        for _, row in y_true_df.iterrows():
            task = row['task']
            trial = row['trial']
            error_onset = row['error_onset'].total_seconds() - 1  # 添加1秒容差
            error_offset = row['error_offset'].total_seconds() + 1  # 添加1秒容差

            # 检查预测的错误是否与实际错误重叠
            detected_err = pos_pred[(pos_pred['task'] == task) & (pos_pred['trial'] == trial) & 
                                    ((pos_pred['start'] >= error_onset) & (pos_pred['start'] <= error_offset)) |
                                    ((pos_pred['end']   >= error_onset) & (pos_pred['end']   <= error_offset)) |
                                    ((pos_pred['start'] <= error_onset) & (pos_pred['end'] >= error_offset))]
            pos_pred.loc[pos_pred['id'].isin(detected_err['id']), 'overlap_error'] = 1
            if len(detected_err) > 0: 
                tp += 1

        # 如果预测不与实际错误重叠，则为假阳性
        fp = len(pos_pred.loc[pos_pred['overlap_error'] == 0])
        
        print(f"True Positive: {tp} ({tp / total_errors * 100:.2f}%)")
        print(f"False Positive: {fp}")

        fn = total_errors - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")

    return tp, fp, total_errors

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

class MiniRocketClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, rocket, classifier):
        self.rocket = rocket
        self.classifier = classifier
        self._is_fitted = False
        
    def fit(self, X, y):
        # 直接对rocket进行fit
        print("Fitting MiniRocket transformer...")
        self.rocket.fit(X.reshape(X.shape[0], 1, X.shape[1]))
            
        # 然后转换数据并fit分类器
        X_transformed = self.rocket.transform(X.reshape(X.shape[0], 1, X.shape[1]))
        self.classifier.fit(X_transformed, y)
        self._is_fitted = True
        return self
                
    def predict(self, X):
        if not self._is_fitted:
            raise ValueError("MiniRocketClassifier has not been fitted yet.")
        X_transformed = self.rocket.transform(X.reshape(X.shape[0], 1, X.shape[1]))
        return self.classifier.predict(X_transformed)
                
    def predict_proba(self, X):
        if not self._is_fitted:
            raise ValueError("MiniRocketClassifier has not been fitted yet.")
        X_transformed = self.rocket.transform(X.reshape(X.shape[0], 1, X.shape[1]))
        return self.classifier.predict_proba(X_transformed)

# 数据预处理函数
def preprocess_data(train_data, val_data, test_data):
    # 数据分割
    df_train = train_data.copy()
    df_val = val_data.copy()
    df_test = test_data.copy()

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

    # 转换为float32以减少内存使用
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    y_err_train = df_train['robot_error']
    y_err_val = df_val['robot_error']

    # 使用KNNImputer填充缺失值，比SimpleImputer更精确
    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    # 使用SMOTE过采样和RandomUnderSampler欠采样结合，创建更平衡的数据集
    sampling_strategy = 0.8  # 正负样本比例
    over_sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    under_sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    sampling_pipeline = ImbPipeline([
        ('over', over_sampler),
        ('under', under_sampler)
    ])
    X_train_balanced, y_train_balanced = sampling_pipeline.fit_resample(X_train_imputed, y_err_train)

    return X_train_balanced, y_train_balanced, X_val_imputed, X_test_imputed, y_err_val, df_train, df_val, df_test

# 训练不同配置的MiniRocket模型
def train_minirocket_models():
    # 设置模型保存路径
    ensemble_model_path = "./BASE/models/clf_ch1_multi_minirocket_ensemble.pkl"
    os.makedirs(os.path.dirname(ensemble_model_path), exist_ok=True)
    
    # 检查集成模型是否已存在
    if os.path.exists(ensemble_model_path):
        print("加载已训练的集成模型...")
        ensemble_clf = joblib.load(ensemble_model_path)
        return ensemble_clf
    
    print("训练新的多MiniRocket集成模型...")
    
    # 存储所有模型
    minirocket_models = []
    
    # 1. 训练不同窗口大小和步长的MiniRocket模型
    for config, train_path in train_data_paths.items():
        print(f"\n处理配置: {config}")
        
        # 加载数据
        train_data_config = pd.read_csv(train_path, low_memory=False)
        val_data_config = pd.read_csv(val_data_paths[config], low_memory=False)
        test_data_config = pd.read_csv(test_data_paths[config], low_memory=False)
        
        # 预处理数据
        X_train_balanced, y_train_balanced, _, _, _, _, _, _ = preprocess_data(
            train_data_config, val_data_config, test_data_config
        )
        
        # 为每个配置训练不同内核数量的MiniRocket模型
        for num_kernels in [2000, 5000, 10000]:
            model_name = f"minirocket_{config}_k{num_kernels}"
            model_path = f"./BASE/models/clf_ch1_{model_name}.pkl"
            
            if os.path.exists(model_path):
                print(f"   从磁盘加载{model_name}模型...")
                minirocket, classifier = joblib.load(model_path)
            else:
                print(f"   训练{model_name}模型...")
                
                # 创建MiniRocket模型
                minirocket = MiniRocketWithProgress(
                    num_kernels=num_kernels,
                    random_state=42
                )
                
                # 将数据转换为MiniRocket需要的格式
                X_train_reshaped = X_train_balanced.reshape(X_train_balanced.shape[0], 1, X_train_balanced.shape[1])
                
                # 训练MiniRocket变换器
                minirocket.fit(X_train_reshaped)
                
                # 提取特征
                X_train_rocket = minirocket.transform(X_train_reshaped)
                
                # 使用不同的分类器
                # 只用这个才能支持软分组
                classifier = LogisticRegressionCV(
                    Cs=10,
                    cv=3,
                    class_weight='balanced',
                    max_iter=2000,
                    random_state=42,
                    n_jobs=-1
                )
        
                # 训练分类器
                with tqdm(desc=f"{model_name} classifier training") as pbar:
                    classifier.fit(X_train_rocket, y_train_balanced)
                    pbar.update(1)
                
                # 保存模型
                joblib.dump((minirocket, classifier), model_path)
                
                # 释放内存
                del X_train_reshaped, X_train_rocket
                gc.collect()
            
            # 创建MiniRocketClassifier包装器
            minirocket_classifier = MiniRocketClassifier(minirocket, classifier)
            
            # 添加到模型列表
            minirocket_models.append((model_name, minirocket_classifier))
    
    # 2. 创建集成模型
    print("\n创建最终集成模型...")
    
    # 设置模型权重 - 根据窗口大小和内核数量分配权重
    weights = []
    for model_name, _ in minirocket_models:
        if "w5_s0.5" in model_name:
            # 窗口大小5，步长0.5的模型权重最高
            if "k10000" in model_name:
                weights.append(0.25)  # 最高权重
            elif "k5000" in model_name:
                weights.append(0.20)
            else:
                weights.append(0.15)
        elif "w3_s0.5" in model_name:
            # 窗口大小3，步长0.5的模型次高权重
            if "k10000" in model_name:
                weights.append(0.15)
            elif "k5000" in model_name:
                weights.append(0.10)
            else:
                weights.append(0.05)
        else:
            # 其他模型较低权重
            if "k10000" in model_name:
                weights.append(0.05)
            elif "k5000" in model_name:
                weights.append(0.03)
            else:
                weights.append(0.02)
    
    # 创建VotingClassifier
    ensemble_clf = VotingClassifier(
        estimators=minirocket_models,
        voting='soft',
        weights=weights
    )
    
    # 使用主配置的数据训练集成模型
    X_train_balanced, y_train_balanced, _, _, _, _, _, _ = preprocess_data(
        train_data, val_data, test_data
    )
    
    # 训练集成模型
    print("训练最终集成模型...")
    ensemble_clf.fit(X_train_balanced, y_train_balanced)
    
    # 保存集成模型
    joblib.dump(ensemble_clf, ensemble_model_path)
    print("集成模型训练并保存完成.")
    
    return ensemble_clf

# 主函数
def main():
    # 预处理主配置的数据
    X_train_balanced, y_train_balanced, X_val_imputed, X_test_imputed, y_err_val, df_train, df_val, df_test = preprocess_data(
        train_data, val_data, test_data
    )
    
    # 训练或加载集成模型
    ensemble_clf = train_minirocket_models()
    
    # 预测验证集
    print("在验证集上进行预测...")
    # scores = ensemble_clf.decision_function(X_val_imputed)
    # probs_err = expit(scores)
    probs_err = ensemble_clf.predict_proba(X_val_imputed)[:, 1]
    
    # 寻找最佳阈值 - 基于F1分数
    thresholds = np.linspace(0.3, 0.7, 20)
    best_f1 = 0
    best_threshold = 0.5
    best_tp = 0
    best_fp = float('inf')

    for threshold in thresholds:
        pred = (probs_err >= threshold).astype(int)
        f1 = f1_score(y_err_val, pred)
        
        # 计算TP和FP
        df_val['y_pred_err'] = pred
        y_pred_df = df_val[['task', 'trial', 'start', 'end', 'y_pred_err']]
        val_trials = df_val['trial'].astype(str).unique()
        y_true_df = all_robot_errors[['task', 'trial', 'error_onset', 'error_offset']].loc[all_robot_errors['trial'].isin(val_trials)]
        tp, fp, total_error = evaluate_error(y_pred_df, y_true_df)
        
        # 更新最佳阈值 - 优先考虑F1分数，其次是TP和FP
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_tp = tp
            best_fp = fp
        elif f1 == best_f1 and tp > best_tp and fp <= best_fp:
            # 如果F1相同，但TP更高且FP不增加，则更新
            best_threshold = threshold
            best_tp = tp
            best_fp = fp

    print(f"最佳阈值: {best_threshold:.4f} (F1: {best_f1:.4f}, TP: {best_tp}, FP: {best_fp})")
    threshold = best_threshold
    
    # 使用最佳阈值进行预测
    pred_err = (probs_err >= threshold).astype(int)
    
    # 预测测试集
    print("在测试集上进行预测...")
    probs_reaction_test = ensemble_clf.predict_proba(X_test_imputed)[:, 1]
    pred_reaction_test = (probs_reaction_test >= threshold).astype(int)

    # 计算评估指标
    f1_e = f1_score(y_err_val, pred_err, average='macro')
    acc_e = accuracy_score(y_err_val, pred_err)
    if sum(y_err_val) > 0: 
        tn, fp, fn, tp = confusion_matrix(y_err_val, pred_err).ravel()
    else: 
        tn, fp, fn, tp = 0, 0, 0, 0

    tpr_e = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr_e = fp / (fp + tn) if (fp + tn) > 0 else 0
    tnr_e = tn / (tn + fp) if (tn + fp) > 0 else 0
    fnr_e = fn / (fn + tp) if (fn + tp) > 0 else 0
    auc_e = roc_auc_score(y_err_val, probs_err)

    # 打印评估指标
    print(
        f"AUC: {auc_e:.4f}, " 
        f"F1: {f1_e:.4f}, Acc: {acc_e:.4f}, "
        f"TPR: {tpr_e:.4f}, FPR: {fpr_e:.4f}, "
        f"TNR: {tnr_e:.4f}, FNR: {fnr_e:.4f}, "
    )

    df_val['y_pred_err'] = pred_err
    df_test['y_pred_reaction'] = pred_reaction_test
    val_trials = df_val['trial'].astype(str).unique()

    # 保存测试集预测结果
    pred_test_df = df_test[['task', 'trial', 'start', 'end', 'y_pred_reaction']]
    pred_test_df.to_csv("./BASE/result/y_pred_ch1_w5_s0.5_multi_minirocket_ensemble.csv", index=False)

    # 分别取预测标签和真实标签
    y_pred_df = df_val[['task', 'trial', 'start', 'end', 'y_pred_err']]
    y_true_df = all_robot_errors[['task', 'trial', 'error_onset', 'error_offset']].loc[all_robot_errors['trial'].isin(val_trials)]

    # 评估模型
    tp, fp, total_error = evaluate_error(y_pred_df, y_true_df)

    # 输出模型性能
    print("\n多MiniRocket集成模型性能:")
    print(f"True Positive Rate: {tp/total_error*100:.2f}%")
    print(f"False Positive: {fp}")
    print(f"F1 Score: {f1_e:.4f}")
    print(f"AUC: {auc_e:.4f}")

if __name__ == "__main__":
    main()
