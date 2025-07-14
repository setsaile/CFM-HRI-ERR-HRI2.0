# 他们的数据处理方式
import os
import pandas as pd
import numpy as np
from datetime import timedelta
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Base Path to Data
# # 测试集地址
BASE_PATH = "./"
# Window Size and Stride
#保证没有重叠的部分
WINDOW_SIZE = 5 # seconds
STRIDE = 1 # seconds

# Data Output File
# 将test生成相应格式
OUTPUT_FILE_TRAIN = f".csv"
OUTPUT_FILE_VAL = f".csv"

# Naming Patterns for Tasks
SYSTEMS = {
    "voice_assistant": ["medical", "trip", "police"],
    "social_robot": ["survival", "discussion"]
}
# SYSTEMS = {
#     "voice_assistant": ["medical"],
#     "social_robot": ["survival"]
# }

# Naming Patterns for Feature Files
FILE_PATTERNS = {
    "face": "{trial}-{task}-openface-output.csv",
    "audio": "{trial}-{task}-eGeMAPSv02-features.csv",
    "transcript": "transcript-{trial}-{task}_embeddings.csv"
}

# Naming Pattern for Label Files
LABEL_FILE_PATTERNS = {
    "robot_errors": "challenge1_robot_error_labels_{task}_train.csv",
    "human_reactions_ch1": "challenge1_user_reaction_labels_{task}_train.csv",
    "human_reactions_ch2": "challenge2_user_reaction_labels_{task}_train.csv"
}

# 修正格式：HH:MM:SS:MS->HH:MM:SS.MS
def fix_timestamp_format(ts):
    if isinstance(ts, str):
        parts = ts.split(':')
        if len(parts) == 4:
            return ':'.join(parts[:3]) + '.' + parts[3]
    return ts

# 脸部信息测试集时间变成HH:MM:SS.MS形式
def face_fix_timestamp_format(ts):
    if isinstance(ts, (int, float, str)):
        try:
            ts_float = float(ts)
            total_seconds = int(ts_float)
            milliseconds = int(round((ts_float - total_seconds) * 1000))

            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60

            return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        except ValueError:
            pass    # 类型不是数字或无法转换时，返回原值
    return ts  

# 音频信息测试集时间变成HH:MM:SS.MS形式
def audio_fix_timestamp_format(ts):
    if isinstance(ts, str) and 'days' in ts:
        try:
            time_part = ts.split('days')[-1].strip()  # 获取 "00:05:30.780000"
            h, m, s = time_part.split(':')
            seconds, microseconds = s.split('.')
            milliseconds = int(round(int(microseconds[:6]) / 1000))
            return f"{int(h):02d}:{int(m):02d}:{int(seconds):02d}.{milliseconds:03d}"
        except Exception:
            pass  # 非字符串或格式不对则返回原值
    return ts  


# 标签存储方式dict{"challenge":content}
def load_and_preprocess_labels(task,is_test):
    """Load all label files for a specific task and preprocess timestamps"""
    labels = {}    
    if is_test==False:
        try:
            # Load robot error labels
            robot_errors = pd.read_csv(
                f"{BASE_PATH}/labels_train/challenge1_train/{LABEL_FILE_PATTERNS['robot_errors'].format(task=task)}"
            )
            robot_errors['error_onset'] = robot_errors['error_onset'].apply(fix_timestamp_format)
            robot_errors['error_onset'] = pd.to_timedelta(robot_errors['error_onset'])
            robot_errors['error_offset'] = robot_errors['error_offset'].apply(fix_timestamp_format)
            robot_errors['error_offset'] = pd.to_timedelta(robot_errors['error_offset'])
            robot_errors['trial_num'] = robot_errors['trial_name'].apply(lambda s: s.split('-', 1)[0])
            labels['robot_errors'] = robot_errors        
            
            # Load human reaction labels
            human_reactions_ch1 = pd.read_csv(
                f"{BASE_PATH}/labels_train/challenge1_train/{LABEL_FILE_PATTERNS['human_reactions_ch1'].format(task=task)}"
            )
            human_reactions_ch1['reaction_onset'] = human_reactions_ch1['reaction_onset'].apply(fix_timestamp_format)
            human_reactions_ch1['reaction_onset'] = pd.to_timedelta(human_reactions_ch1['reaction_onset'])
            human_reactions_ch1['reaction_offset'] = human_reactions_ch1['reaction_offset'].apply(fix_timestamp_format)
            human_reactions_ch1['reaction_offset'] = pd.to_timedelta(human_reactions_ch1['reaction_offset'])
            human_reactions_ch1['trial_num'] = human_reactions_ch1['trial_name'].apply(lambda s: s.split('-', 1)[0])
            labels['human_reactions_ch1'] = human_reactions_ch1        
            
            human_reactions_ch2 = pd.read_csv(
                f"{BASE_PATH}/labels_train/challenge2_train/{LABEL_FILE_PATTERNS['human_reactions_ch2'].format(task=task)}"
            )
            human_reactions_ch2['reaction_onset'] = human_reactions_ch2['reaction_onset'].apply(fix_timestamp_format)
            human_reactions_ch2['reaction_onset'] = pd.to_timedelta(human_reactions_ch2['reaction_onset'])
            human_reactions_ch2['reaction_offset'] = human_reactions_ch2['reaction_offset'].apply(fix_timestamp_format)
            human_reactions_ch2['reaction_offset'] = pd.to_timedelta(human_reactions_ch2['reaction_offset'])
            human_reactions_ch2['trial_num'] = human_reactions_ch2['trial_name'].apply(lambda s: s.split('-', 1)[0])
            labels['human_reactions_ch2'] = human_reactions_ch2    
        
        except Exception as e:
            print(f"Error loading label files for task {task}: {str(e)}")
            return None, None, None, None  
        # labels包含三个error形式,robot_errors是第一个错误形式,human_reactions_ch1是机器反应的第一个错误形式,human_reactions_ch2是第二个错误形式
        return labels, robot_errors, human_reactions_ch1, human_reactions_ch2
    else:
        return None, None, None, None

# 标记label中err(逻辑是：区间只要有重叠部分就标记为1,没有具体计算重叠的程度,可能会使得标签1变多)
def is_error_window(start_td, end_td, trial_num, labels_robot_errors, labels_human_reactions_ch1, labels_human_reactions_ch2):

    """Check if window contains any labels"""
    labels = {
        'robot_error': 0,
        'reaction_ch1': 0,
        'reaction_ch2': 0,
        'reaction_type': None
    }    
    # if labels_robot_errors is not None:
    # # Check robot errors
    #     trial_errors = labels_robot_errors[labels_robot_errors['trial_num'] == trial_num]
    #     for _, row in trial_errors.iterrows():
    #         # 只要有重叠的部分,就将这个时间段设置为1(没有考虑重叠的覆盖度有多少)
    #         if (row['error_onset'] <= start_td and row['error_offset'] >= start_td) or (row['error_onset'] <= end_td and row['error_offset'] >= end_td):
    #             labels['robot_error'] = 1
    #             break    
    if labels_robot_errors is not None:
        # Check robot errors
        trial_errors = labels_robot_errors[labels_robot_errors['trial_num'] == trial_num]
        
        def to_seconds(time_value):
            if isinstance(time_value, (pd.Timedelta, pd._libs.tslibs.timedeltas.Timedelta)):
                return time_value.total_seconds()
            elif hasattr(time_value, 'total_seconds'):
                return time_value.total_seconds()
            else:
                return float(time_value)
        start_seconds = to_seconds(start_td)
        end_seconds = to_seconds(end_td)
        # 计算当前时间段的总长度（秒）
        segment_duration = end_seconds - start_seconds
        total_overlap_duration = 0
        # error_types = []
        for _, row in trial_errors.iterrows():
            # 统一error时间的数据类型
            error_start = to_seconds(row['error_onset'])
            error_end = to_seconds(row['error_offset'])
            
            # 计算重叠区间
            overlap_start = max(start_seconds, error_start)
            overlap_end = min(end_seconds, error_end)
            
            # 如果有重叠
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                total_overlap_duration += overlap_duration
                # error_types.append(row['error_id'])
        # 计算重叠比例
        overlap_ratio = total_overlap_duration / segment_duration if segment_duration > 0 else 0
        # 只有当重叠比例 >= 50% 时才设置标签为1
        if overlap_ratio >= 0.5:
            labels['robot_error'] = 1
            # 如果有多个error类型，可以选择最频繁的或者合并
            # if error_types:
            #     # 选择最频繁出现的error_id
            #     from collections import Counter
            #     most_common_type = Counter(error_types).most_common(1)[0][0]
            #     labels['error_id'] = most_common_type
                
    if labels_human_reactions_ch1 is not None:
        # Check human reactions (Challenge 1)
        # 下面是baseline的逻辑
        trial_reactions_ch1 = labels_human_reactions_ch1[labels_human_reactions_ch1['trial_num'] == trial_num]
        for _, row in trial_reactions_ch1.iterrows():
            if (row['reaction_onset'] <= start_td and row['reaction_offset'] >= start_td) or (row['reaction_onset'] <= end_td and row['reaction_offset'] >= end_td):
                labels['reaction_ch1'] = 1
                break    
            
    # if labels_human_reactions_ch2 is not None:
    # # Check human reactions (Challenge 2)
    #     trial_reactions_ch2 = labels_human_reactions_ch2[labels_human_reactions_ch2['trial_num'] == trial_num]
    #     for _, row in trial_reactions_ch2.iterrows():
    #         # 将端点被重复取到的可能去除
    #         if (row['reaction_onset'] <= start_td and row['reaction_offset'] > start_td) or (row['reaction_onset'] < end_td and row['reaction_offset'] >= end_td):
    #             labels['reaction_ch2'] = 1
    #             labels['reaction_type'] = row['reaction_type']
    #             break    
    
    # 参考之前标签选取众数的形式进行设计标签2   
    if labels_human_reactions_ch2 is not None:
        # Check human reactions (Challenge 2)
        trial_reactions_ch2 = labels_human_reactions_ch2[labels_human_reactions_ch2['trial_num'] == trial_num]
        
        # 统一时间数据类型 - 转换为秒数进行计算
        def to_seconds(time_value):
            if isinstance(time_value, (pd.Timedelta, pd._libs.tslibs.timedeltas.Timedelta)):
                return time_value.total_seconds()
            elif hasattr(time_value, 'total_seconds'):  # datetime.timedelta
                return time_value.total_seconds()
            else:
                return float(time_value)
        
        start_seconds = to_seconds(start_td)
        end_seconds = to_seconds(end_td)
        
        # 计算当前时间段的总长度（秒）
        segment_duration = end_seconds - start_seconds
        
        # 计算与所有reaction的重叠时间总和
        total_overlap_duration = 0
        reaction_types = []
        
        for _, row in trial_reactions_ch2.iterrows():
            # 统一reaction时间的数据类型
            reaction_start = to_seconds(row['reaction_onset'])
            reaction_end = to_seconds(row['reaction_offset'])
            
            # 计算重叠区间
            overlap_start = max(start_seconds, reaction_start)
            overlap_end = min(end_seconds, reaction_end)
            
            # 如果有重叠
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                total_overlap_duration += overlap_duration
                reaction_types.append(row['reaction_type'])
        
        # 计算重叠比例
        overlap_ratio = total_overlap_duration / segment_duration if segment_duration > 0 else 0
        
        # 只有当重叠比例 >= 50% 时才设置标签为1
        if overlap_ratio >= 0.5:
            labels['reaction_ch2'] = 1
            # 如果有多个reaction类型，可以选择最频繁的或者合并
            if reaction_types:
                # 选择最频繁出现的reaction_type
                from collections import Counter
                most_common_type = Counter(reaction_types).most_common(1)[0][0]
                labels['reaction_type'] = most_common_type
    
    return labels

# 提取face,audio,script按照窗口划分,有mean,min,max,std四个形式,标注出是否含有err,返回df
def extract_windows(face_df, audio_df, transcript_df, trial_name, labels_robot_errors, labels_human_reactions_ch1, labels_human_reactions_ch2, window_size, stride, is_test):
    
    """Extract time windows with features and labels, flattening agg_ vectors into columns."""
    # print(f"      Extracting windows for: {trial_name}")
    # print(f"      Face max: {face_df['timestamp'].max()}")
    # print(f"      Audio end max: {audio_df['end'].max()}")
    # print(f"      Transcript end max: {transcript_df['end'].max()}")

    rows = []
    win_delta = timedelta(seconds=window_size)
    str_delta = timedelta(seconds=stride)

    task_duration = max(
        face_df['timestamp'].max(),
        audio_df['end'].max(),
        transcript_df['end'].max()
    )   # 取三者中最大的为task的总长度

    start = timedelta(seconds=0)
    # 对原始数据进行切片合并
    while start + win_delta <= task_duration:
        end = start + win_delta

        # 取窗口大小的切片(fw是时间戳在start和end的内部,aw和tw是有交集就取出来)
        fw = face_df[(face_df['timestamp'] >= start) & (face_df['timestamp'] < end)]
        aw = audio_df[(audio_df['start'] < end) & (audio_df['end'] > start)]
        tw = transcript_df[(transcript_df['start'] < end) & (transcript_df['end'] > start)]

        # 去除非特征列,分别使用mean,min,max,std取值来代替这个时间段内特征的取值
        if not fw.empty:
            agg_face_mean = fw.drop(columns=['frame','face_id','timestamp']).mean().add_suffix('_mean')
            agg_face_min = fw.drop(columns=['frame','face_id','timestamp']).min().add_suffix('_min')
            agg_face_max = fw.drop(columns=['frame','face_id','timestamp']).max().add_suffix('_max')
            agg_face_std = fw.drop(columns=['frame','face_id','timestamp']).std().add_suffix('_std')
        else:
            cols_mean = face_df.drop(columns=['frame','face_id','timestamp']).add_suffix('_mean').columns
            agg_face_mean = pd.Series(0, index=cols_mean)

            cols_min = face_df.drop(columns=['frame','face_id','timestamp']).add_suffix('_min').columns
            agg_face_min = pd.Series(0, index=cols_min)

            cols_max = face_df.drop(columns=['frame','face_id','timestamp']).add_suffix('_max').columns
            agg_face_max = pd.Series(0, index=cols_max)

            cols_std = face_df.drop(columns=['frame','face_id','timestamp']).add_suffix('_std').columns
            agg_face_std = pd.Series(0, index=cols_std)

        if not aw.empty:
            agg_audio_mean = aw.drop(columns=['start','end']).mean().add_suffix('_mean')
            agg_audio_min = aw.drop(columns=['start','end']).min().add_suffix('_min')
            agg_audio_max = aw.drop(columns=['start','end']).max().add_suffix('_max')
            agg_audio_std = aw.drop(columns=['start','end']).std().add_suffix('_std')
        else:
            cols_mean = audio_df.drop(columns=['start','end']).add_suffix('_mean').columns
            agg_audio_mean = pd.Series(0, index=cols_mean)

            cols_min = audio_df.drop(columns=['start','end']).add_suffix('_min').columns
            agg_audio_mean = pd.Series(0, index=cols_min)

            cols_max = audio_df.drop(columns=['start','end']).add_suffix('_max').columns
            agg_audio_max = pd.Series(0, index=cols_max)

            cols_std = audio_df.drop(columns=['start','end']).add_suffix('_std').columns
            agg_audio_std = pd.Series(0, index=cols_std)


        if not tw.empty:
            agg_transcript_mean = tw.drop(columns=['start','end','confidence','speaker','word_count']).mean().add_suffix('_mean')
            agg_transcript_min = tw.drop(columns=['start','end','confidence','speaker','word_count']).mean().add_suffix('_min')
            agg_transcript_max = tw.drop(columns=['start','end','confidence','speaker','word_count']).mean().add_suffix('_max')
            agg_transcript_std = tw.drop(columns=['start','end','confidence','speaker','word_count']).mean().add_suffix('_std')
        else:
            cols_mean = transcript_df.drop(columns=['start','end','confidence','speaker','word_count']).add_suffix('_mean').columns
            agg_transcript_mean = pd.Series(0, index=cols_mean)

            cols_min = transcript_df.drop(columns=['start','end','confidence','speaker','word_count']).add_suffix('_min').columns
            agg_transcript_min = pd.Series(0, index=cols_min)

            cols_max = transcript_df.drop(columns=['start','end','confidence','speaker','word_count']).add_suffix('_max').columns
            agg_transcript_max = pd.Series(0, index=cols_max)

            cols_std = transcript_df.drop(columns=['start','end','confidence','speaker','word_count']).add_suffix('_std').columns
            agg_transcript_std = pd.Series(0, index=cols_std)

        # 测试集就少了label的列,只剩下start和end这两列
        if is_test:
            row={
                'start': start.total_seconds(),
                'end':   end.total_seconds(),
            }
        else:    
        # labels
            labels = is_error_window(start, end, trial_name,
                                    labels_robot_errors,
                                    labels_human_reactions_ch1,
                                    labels_human_reactions_ch2)

        # build flat row
            row = {
                'start': start.total_seconds(),
                'end':   end.total_seconds(),
                'robot_error': labels['robot_error'],
                'reaction_ch1': labels['reaction_ch1'],
                'reaction_ch2': labels['reaction_ch2'],
                'reaction_type': labels['reaction_type'],
            }
        # prefix & merge each agg_ series
        row.update(agg_face_mean.add_prefix('face_').to_dict())
        row.update(agg_audio_mean.add_prefix('audio_').to_dict())
        row.update(agg_transcript_mean.add_prefix('transcript_').to_dict())

        row.update(agg_face_min.add_prefix('face_').to_dict())
        row.update(agg_audio_min.add_prefix('audio_').to_dict())
        row.update(agg_transcript_min.add_prefix('transcript_').to_dict())

        row.update(agg_face_max.add_prefix('face_').to_dict())
        row.update(agg_audio_max.add_prefix('audio_').to_dict())
        row.update(agg_transcript_max.add_prefix('transcript_').to_dict())

        row.update(agg_face_std.add_prefix('face_').to_dict())
        row.update(agg_audio_std.add_prefix('audio_').to_dict())
        row.update(agg_transcript_std.add_prefix('transcript_').to_dict())

        rows.append(row)
        start += str_delta

    return pd.DataFrame(rows)

def process_all_data(WINDOW_SIZE, STRIDE, is_test):
    all_windows = []
    
    for system, tasks in SYSTEMS.items():
        print(f"\nProcessing system: {system.upper()}")        
        
        for task in tasks:
            print(f"\n  Task: {task}")            
            
            # Load label files for this task
            labels, robot_errors, human_reactions_ch1, human_reactions_ch2 = load_and_preprocess_labels(task,is_test=False)
            # labels,robot_errors, human_reactions_ch1, human_reactions_ch2 = None, None, None, None
            print("labels, robot_errors, human_reactions_ch1, human_reactions_ch2")
            print(labels, robot_errors, human_reactions_ch1, human_reactions_ch2)
            # labels = None
            # if labels is None:
            #     continue 

            # 训练集生成    
            face_files = glob(f"{BASE_PATH}/face_head_features_train/{system}/{task}/*-{task}-openface-output.csv")
            audio_files = glob(f"{BASE_PATH}/audio_features_train/{system}/{task}/*-{task}-eGeMAPSv02-features.csv")
            transcript_files = glob(f"{BASE_PATH}/transcript_features_train/{system}/{task}/transcript-*-{task}_embeddings.csv") 
            
            # # 测试集生成
            # face_files = glob(f"{BASE_PATH}/face_head_features_test/{system}/{task}/*-{task}-openface-output.csv")     
            # audio_files = glob(f"{BASE_PATH}/audio_features_test/{system}/{task}/*-{task}-eGeMAPSv02-features.csv")   
            # transcript_files = glob(f"{BASE_PATH}/transcript_features_test/{system}/{task}/transcript-*-{task}_embeddings.csv")   
            
            print(f"    Found {len(face_files)} face files")
            print(f"    Found {len(audio_files)} audio files")
            print(f"    Found {len(transcript_files)} transcript files")    
    
            # Process each trial
            for face_file in sorted(face_files):
                trial_name = os.path.basename(face_file).split('-')[0]
                print(f"    Processing trial: {trial_name}")             
    
                try:
                    # Find matching files
                    matching_audio = [f for f in audio_files if os.path.basename(f).startswith(trial_name)]
                    matching_transcript = [f for f in transcript_files if f"transcript-{trial_name}-{task}" in os.path.basename(f)]                    
                    
                    if not matching_audio or not matching_transcript:
                        print(f"      Missing matching files")
                        continue         
    
                    # Load face data
                    try:
                        face_df = pd.read_csv(face_file)
                        face_df.columns = face_df.columns.str.lstrip()
                        if face_df.empty:
                            print(f"      Skipping empty face file: {face_file}")
                            continue
                    except Exception as e:
                        print(f"      Could not read face file: {face_file} — {e}")
                        continue           
    
                    # Check for timestamp column
                    if 'timestamp' not in face_df.columns:
                        print(f"      :x: 'timestamp' column missing in: {face_file}")
                        continue
    
                    # Load audio data
                    audio_df = pd.read_csv(matching_audio[0])
                    audio_df.columns = audio_df.columns.str.strip()
    
                    # Load transcript data
                    transcript_df = pd.read_csv(matching_transcript[0])
                    transcript_df.columns = transcript_df.columns.str.strip()                    
                    
                    # Fix timestamps
                    #pd.to_timedelta 使时间可用于后续计算
                    # 给的face和audio的测试集处理方式不同
                    if is_test: 
                        face_df['timestamp'] = face_df['timestamp'].apply(face_fix_timestamp_format)
                        face_df['timestamp'] = pd.to_timedelta(face_df['timestamp'])                    
    
                    else:
                        face_df['timestamp'] = face_df['timestamp'].apply(fix_timestamp_format)
                        face_df['timestamp'] = pd.to_timedelta(face_df['timestamp'])                    

                    if is_test:
                        audio_df['start'] = audio_df['start'].apply(audio_fix_timestamp_format)
                        audio_df['start'] = pd.to_timedelta(audio_df['start'])
                        audio_df['end'] = audio_df['end'].apply(audio_fix_timestamp_format)
                        audio_df['end'] = pd.to_timedelta(audio_df['end'])                       
                    else:
                        audio_df['start'] = audio_df['start'].apply(fix_timestamp_format)
                        audio_df['start'] = pd.to_timedelta(audio_df['start'])
                        audio_df['end'] = audio_df['end'].apply(fix_timestamp_format)
                        audio_df['end'] = pd.to_timedelta(audio_df['end'])                    
                        
                    transcript_df['start'] = transcript_df['start'].apply(fix_timestamp_format)
                    transcript_df['start'] = pd.to_timedelta(transcript_df['start'])
                    transcript_df['end'] = transcript_df['end'].apply(fix_timestamp_format)
                    transcript_df['end'] = pd.to_timedelta(transcript_df['end'])                    
    
                    # Extract windows
                    windows = extract_windows(face_df, audio_df, transcript_df, trial_name, robot_errors, human_reactions_ch1, human_reactions_ch2, WINDOW_SIZE, STRIDE,is_test=False)                    
                        
                    # Add metadata
                    windows['system'] = system
                    windows['task'] = task
                    windows['trial'] = trial_name                    
                    
                    all_windows.append(windows)
                    print(f"      Extracted {len(windows)} windows")                
                
                except Exception as e:
                    print(f"      Error processing trial: {str(e)}")
                    continue    
    # print("all_windows:", all_windows)
    final_df = pd.concat(all_windows, ignore_index=True)
    print(f"\nTOTAL WINDOWS EXTRACTED: {len(final_df)}")
    return final_df

def main():
    windows = process_all_data(WINDOW_SIZE, STRIDE, is_test = False)
    # windows.to_csv(OUTPUT_FILE, index=False)
    # 如果是生成测试集直接将windows写入即可
    
    
    # 下面操作划分训练集和验证集
    # 获取唯一用户列表（每个用户由 task + trial 唯一表示）
    users = windows[['task', 'trial']].drop_duplicates()
    
    # 划分为训练用户和验证用户
    train_users, val_users = train_test_split(
        users,
        test_size=9,        # 9 个用户为验证集
        random_state=42,    # 固定随机种子
        shuffle=True
    )

    # 把训练用户和验证用户组合成列表用于过滤
    train_user_set = set([tuple(x) for x in train_users.values])
    val_user_set = set([tuple(x) for x in val_users.values])

    # 根据划分结果筛选原始行，保持原始顺序不变
    train_df = windows[windows[['task', 'trial']].apply(tuple, axis=1).isin(train_user_set)]
    val_df   = windows[windows[['task', 'trial']].apply(tuple, axis=1).isin(val_user_set)]

    # 保存为 CSV 文件
    train_df.to_csv(OUTPUT_FILE_TRAIN, index=False)
    val_df.to_csv(OUTPUT_FILE_VAL, index=False)

if __name__ == "__main__":
    main()

