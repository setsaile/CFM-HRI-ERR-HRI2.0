import math
import os
import re

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm
import random

class DataLoader_HRI2:
    """
    加载输入数据和标签

    属性：
        data_dir: 数据存储目录
        verbose: 如果为 True, 则打印调试信息
        val_X: 包含取决于所选折叠的验证数据
        val_Y: 包含取决于所选折叠的验证标签
        train_Y: 包含取决于所选折叠的训练标签
        train_X: 包含取决于所选折叠的训练数据
        test_X: 包含测试数据（仅供比赛组织者使用）
        test_Y: 包含测试标签（仅供比赛组织者使用）
        all_X: 包含所有数据（训练集 + 验证集）
        all_Y: 包含所有标签（训练集 + 验证集）
        fold_info: 包含折叠信息（哪些会话编号属于哪个折叠）
    """
    def __init__(self, data_dir: str = "./", verbose: bool = False, find_csv = True):

        self.data_dir = data_dir
        self.verbose = verbose
        # 验证集和对应的两个挑战的label
        self.val_X = []
        self.val_Y1 = []
        self.val_Y2 = []
        self.val_Y = []
        
        self.train_X = []
        self.train_Y = []
        self.train_Y1 = []
        self.train_Y2 = []
        # 拿到测试集再处理这个
        self.test_X = []
        self.test_Y = []

        self.all_X = []
        self.all_Y1 = []    # Y1是challenge1的标签
        self.all_Y2 = []    # Y2是challenge2的标签
        # true就是直接取处理好的数据
        if find_csv: 
            merge_df = pd.read_csv('BASEdata/merged_df.csv')
            transcript_df = pd.read_csv('BASEdata/transcript.csv')
            challenge1 = pd.read_csv('BASEdata/challenge1.csv')
            challenge2 = pd.read_csv('BASEdata/challenge2.csv')
            self.all_X.append((merge_df, transcript_df))
            self.all_Y1.append(challenge1)
            self.all_Y2.append(challenge2)
            # 查看数据集前10行内容,以及长度
            print("\n=== merge_list 前10行 ===")
            print(merge_df.head(10))
            print(f"\nmerge_list 长度: {len(merge_df)}")
            # 查看一下有多少个不同的场景
            print('有多少个不重复的场景:',len(merge_df['trial_name'].unique()))

            print("\n=== transcript_list 前10行 ===")
            print(transcript_df.head(10))
            print(f"\ntranscript_list 长度: {len(transcript_df)}")
            print('有多少个不重复的场景:',len(transcript_df['trial_name'].unique()))

            print("\n=== self.all_Y1 前10行 ===")
            print(self.all_Y1[0].head(10))
            print(f"\nself.all_Y1 长度: {len(self.all_Y1[0])}")
            print('有多少个不重复的场景:',len(self.all_Y1[0]['trial_name'].unique()))
            print('有多少个0,1标签:', self.all_Y1[0]['challenge1'].value_counts().to_dict())

            print("\n=== self.all_Y2 前10行 ===")
            print(self.all_Y2[0].head(10))
            print(f"\nself.all_Y2 长度: {len(self.all_Y2[0])}")
            print('有多少个不重复的场景:',len(self.all_Y2[0]['trial_name'].unique()))
            print('有多少个0,1标签:', self.all_Y2[0]['challenge2'].value_counts().to_dict())
            self.fold_info = self.load_fold_info(merge_df)

        # false就是从头开始处理数据
        if not find_csv:
            # 加载face、audio、transcript三者数据，将每个模态对应的session写到写到第一列中区分
            # 将数据暂时存为元祖类型:eg:[('file1_train.csv', <DataFrame1>),
            # ('file2_val.csv',   <DataFrame2>),]
            # .csv文件中的第一列是trial_name: 实验者编号 + 实验的session
            face_head_data = self.load_data(data_dir + 'face_head_features_train/')
            audio_data = self.load_data(data_dir + 'audio_features_train/')
            transcript_data = self.load_data(data_dir + 'transcript_features_train/')

            # 加载label,分别返回两个挑战的label
            challenge1,challenge2 = self.load_labels(data_dir + 'labels_train/')

            # 将时间段的数据集转换成时间戳形式
            # 对齐100帧的采样频率
            audio_data = self.process_audio(audio_data)
            face_head_data = self.process_face(face_head_data)
            transcript_data = self.process_CLIP(transcript_data)

            # # 查看一下信息
            # print("face_head:")
            # print(face_head_data[2][1].head(3))
            # print(len(face_head_data[2][1]))
            # print("audio:")
            # print(audio_data[2][1].head(3))
            # print(len(audio_data[2][1]))
            # print("\trans:")
            # print(transcript_data[2][1].head(5))
            # print(len(transcript_data[2][1]))

            # 合并模态,三个模态合并在一起(face多了一个样本,使用audio_data正好可以去掉)
            
            # # 测试
            # first_print = True

            # 合并这音频和图像模态特征
            # merge_df数据类型 [(<merged>,<transrcipt>)]两个都是dataframe
            for filename, _ in tqdm(audio_data, desc="Muilt-Data"):
                merge_df, match_transcript = self.merge_X_data(face_head_data, audio_data,
                                            transcript_data, filename)
                self.all_X.append((merge_df, match_transcript))
                # # 测试
                # if first_print:
                #     df = merge_df[0][1]
                #     df.to_csv('muilt2.csv', index = False)
                #     first_print = False
                #     print(type(merge_df))

            # # 查看标签和feature的长度
            # print('challenge1: ', len(challenge1) )
            # print('challenge2: ', len(challenge2) )
            # print('feature: ', len(self.all_X) )
            # for i, pair in enumerate(self.all_X):
            #     if len(pair) != 2:
            #         print(f"❌ 第 {i} 项不是二元组或长度不为2")
            #     else:
            #         print(f"✅ 第 {i} 项长度正确")

            # 将标签送入all_Y1中去
            for filename, df in challenge1:
                self.all_Y1.append(df)
            for filename, df in challenge2:
                self.all_Y2.append(df)
            
            # 按行axis=0合并Dataframe,此时all_Y都已经是DataFrame
            self.all_Y1 = pd.concat(self.all_Y1)
            self.all_Y2 = pd.concat(self.all_Y2)

            merge_list = [pair[0] for pair in self.all_X]
            transcript_list = [pair[1] for pair in self.all_X]
            merge_list = pd.concat(merge_list)
            transcript_list = pd.concat(transcript_list)

            # 将feature和label都进行列名清理,有空格的删除,方便后文匹配
            self.all_Y1.columns = self.all_Y1.columns.str.replace(' ','', regex = False)
            self.all_Y2.columns = self.all_Y2.columns.str.replace(' ','', regex = False)
            merge_list.columns = merge_list.columns.str.replace(' ','', regex = False)
            transcript_list.columns = transcript_list.columns.str.replace(' ','', regex = False)

            # # 测试：检查是否存在label1、2中有的帧，但是feature中没有的
            # frame1, frame2 = self.check_missing_frames(self.all_Y1, self.all_Y2, merge_list)
            
            # 为label1和label2进行标签填充, 没出现的用0来填充
            self.all_Y1, self.all_Y2 = self.add_0_into_label(self.all_Y1, self.all_Y2, merge_list)
            
            # 查看数据集前10行内容,以及长度
            print("\n=== merge_list 前10行 ===")
            print(merge_list.head(10))
            print(f"\nmerge_list 长度: {len(merge_list)}")
            # 查看一下有多少个不同的场景
            print('有多少个不重复的列:',len(merge_list['trial_name'].unique()))

            print("\n=== transcript_list 前10行 ===")
            print(transcript_list.head(10))
            print(f"\ntranscript_list 长度: {len(transcript_list)}")
            print('有多少个不重复的列:',len(transcript_list['trial_name'].unique()))

            print("\n=== self.all_Y1 前10行 ===")
            print(self.all_Y1.head(10))
            print(f"\nself.all_Y1 长度: {len(self.all_Y1)}")
            print('有多少个不重复的列:',len(self.all_Y1['trial_name'].unique()))
            print('有多少个0,1标签:', self.all_Y1['challenge1'].value_counts().to_dict())

            print("\n=== self.all_Y2 前10行 ===")
            print(self.all_Y2.head(10))
            print(f"\nself.all_Y2 长度: {len(self.all_Y2)}")
            print('有多少个不重复的列:',len(self.all_Y2['trial_name'].unique()))
            print('有多少个0,1标签:', self.all_Y2['challenge2'].value_counts().to_dict())

            # 存入交叉验证的验证集编号,8-fold(加入了随机种子,确保每次划分的值都是一样的)
            self.fold_info = self.load_fold_info(merge_list)

            # 清空列表,重新组合
            self.all_X = []
            self.all_X.append((merge_list, transcript_list))
            self.all_Y1.to_csv('data/challenge1.csv',index = False)
            self.all_Y2.to_csv('data/challenge2.csv',index = False)
            self.all_X[0][0].to_csv('data/merged_df.csv',index = False)
            self.all_X[0][1].to_csv('data/transcript.csv',index = False)

        # print('feature长度:', len(self.all_X))

        # all_X:[(merge_df, transcrip)],all_Y1:[label1],all_Y2:[label2]
                                       
    @staticmethod
    def extract_file_number(filename: str) -> int:
        """
        提取文件名中的数字,找到数字就返回
        """  
        match = re.search(r'\d+', filename)  # 查找文件名中的第一个数字
        return int(match.group()) if match else None  # 如果找到数字，则返回该数字，否则返回 None
    
    # 时间形式格式是:HH:MM:SS:MS,需要转换成float类型
    @staticmethod
    def time_str_to_seconds(time_str):
        h, m, s, ms = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s + ms / 1000
    
    def process_audio(self, df_list):
        """
        处理audio数据集,将时间段转换成时间戳的形式
        """
        # # 用于测试是否变换成功
        # first_csv = True
        
        processed = []
        for filename, df in tqdm(df_list, desc="Processing audio"):
            df = df.copy()
            # 删除end这一列
            if 'end' in df.columns:
                df.drop(columns = ['end'], inplace =True)
            
            # 将start这一列变成frame帧
            if 'start' in df.columns:
                df['start'] = range(1, len(df)+1)
                df.rename(columns = {'start':'frame'}, inplace = True)
            
            # # 测试
            # if first_csv:
            #     df.to_csv('first_audio.csv',index = False)
            #     first_csv = False

            processed.append((filename, df))
        return processed
    
    def process_face(self, df_list):
        """
        上采样脸部头部信息，从 30 帧变成 100 帧。
        对每个DataFrame连续列采用线性插值, 离散列采用最近临近插值
        删除binary AU脸部特征
        """
        # # 测试
        # first_print = True
        processed = []
        target_fps = 100
        origin_fps = 30
        au_cols_to_drop = [
        'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c',
        'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c',
        'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c', 'success', 'timestamp', 'face_id'] 
        
        for filename, df in tqdm(df_list, desc="Processing face"):
            # 先把这些多余的列给删除
            # 去掉前后空格,列表中列名前面有一个空格出现
            df.columns = [col.strip() for col in df.columns]  
            cols_to_drop = [col for col in df.columns if col.lower() in [c.lower() for c in au_cols_to_drop]]
            df = df.drop(columns=cols_to_drop)
            
            total_frames = len(df)
            upsampled_len = int(total_frames * target_fps / origin_fps)
            new_frames = np.linspace(1, total_frames, upsampled_len)
            
            df_resample = pd.DataFrame()
            
            # frame列为整数递增，从1开始
            df_resample['frame'] = np.arange(1, upsampled_len + 1)

            for col in df.columns:
                if col == 'frame':
                    continue
                elif col == 'trial_name':
                    df_resample.insert(0, 'trial_name', df['trial_name'].iloc[0])  # trial_name设为第一列
                # 对剩余的连续信息采用线性插值
                else:   
                    # 获取原始列小数精度
                    sample_val = df[col].dropna().iloc[0]
                    decimals = len(str(sample_val).split('.')[-1]) if '.' in str(sample_val) else 0
                    
                    # 采用线性插值法进行上采样的推算,得到100fps的取值
                    f = interp1d(np.arange(1, total_frames + 1), df[col], kind='linear', fill_value='extrapolate')
                    interpolated = f(new_frames)
                    df_resample[col] = np.round(interpolated, decimals)
            # # 测试
            # if first_print:
            #     df_resample.to_csv('second_face.csv', index=False)
            #     first_print = False

            processed.append((filename, df_resample))
        return processed
    
    # # 将时间片一直抽取重复帧,占用内存比较大
    # def process_CLIP(self, df_list):
    #     """
    #     将CLIP_Embedding嵌入的文本从时间段变成时间戳格式
    #     将给出的时间段部分按照100fps帧率划分,进行相同的填值
    #     没有出现的时间段先暂时不管,最后在融合的时候填充0进行处理
    #     speaker: A->0, B->1
    #     """
    #     # 测试
    #     first_print = True

    #     processed = []
    #     fps = 100
    #     for filename, df in tqdm(df_list, desc="Processing CLIP embeddings"):
    #         rows = []
    #         for _, row in df.iterrows():
    #             start_sec = self.time_str_to_seconds(row['start'])
    #             end_sec = self.time_str_to_seconds(row['end'])

    #             # 时间换算成帧号
    #             start_frame = int(np.floor(start_sec * fps))
    #             end_frame = int(np.floor(end_sec * fps))

    #             # 将特征抽取重复帧进行嵌入
    #             embedding = row[[col for col in df.columns if col.startswith('dim_')]].astype(float).values
    #             speaker = 0 if row['speaker']=='A' else 1
    #             confidence = float(row['confidence'])
    #             word_count = int(row['word_count'])

    #             for frame in range (start_frame, end_frame+1):
    #                 rows.append({
    #                     'trial_name': row['trial_name'],
    #                     'frame': frame,
    #                     'confidence': confidence,
    #                     'speaker': speaker,             
    #                     'word_count': word_count,
    #                     **{f'dim_{i}': embedding[i] for i in range(len(embedding))}
    #                 })
                
    #         df_resample = pd.DataFrame(rows)

    #         # 测试
    #         if first_print:
    #             df_resample.to_csv('second_trans.csv', index = False)
    #             first_print = False

    #         processed.append((filename, df_resample))   
    #     return processed

    # 并不把每条重复帧数据列举出来,而是列出这些重复帧从哪一帧开始,从哪一帧结束
    def process_CLIP(self, df_list):
        """
        将CLIP_Embedding嵌入的文本从时间段变成帧格式
        将start/end时间转成帧号,按100fps重复嵌入特征
        speaker: A->0, B->1
        """
        # first_print = True
        processed = []
        fps = 100

        for filename, df in tqdm(df_list, desc="Processing CLIP embeddings"):
            # 将 start 和 end 时间转换为帧号
            df['start'] = (df['start'].apply(self.time_str_to_seconds) * fps).astype(int)
            df['end'] = (df['end'].apply(self.time_str_to_seconds) * fps).astype(int)
            # 重新命名列名,表示是帧
            df = df.rename(columns={'start': 'start_frame', 'end': 'end_frame'})
            
            # 将 speaker 从 A/B 转换为 0/1
            df['speaker'] = df['speaker'].apply(lambda x: 0 if x == 'A' else 1)

            # # 测试：仅保存转换后的 CSV
            # if first_print:
            #     df.to_csv(f'second_trans.csv', index=False)
            #     first_print = False

            processed.append((filename, df))

        return processed



    def load_data(self, data_dir, verbose = False):
        """
        加载csv数据集,到数据框列表中
        """

        data_frames = []
        # first_saved = True # 用于查看处理结果是否正确

        # 遍历data_dir下所有的文件
        for root, dirs, files in os.walk(data_dir):
            # 在这5个文件夹之下
            if os.path.basename(root) in sorted(['discussion', 'survival', 'medical', 'police', 'trip']):
                # self.extract_file_number 根据文件名称的数字进行排序
                for filename in sorted(files, key = self.extract_file_number):
                    if filename.endswith(".csv"):
                        # 提取 session 名称，如 '1-medical'
                        # transcript 命名故意颠倒顺序了
                        if 'transcript' in data_dir:
                            session_name = filename.split('-')[1] + '-' + os.path.basename(root)
                        else:
                            session_name = filename.split('-')[0] + '-' + os.path.basename(root)

                        # 读取 CSV 文件为 DataFrame
                        df = pd.read_csv(os.path.join(root, filename))
                    
                        # 在 DataFrame 的第一列添加 trial_name
                        df.insert(0, 'trial_name', session_name)

                        # # 保存第一个看看效果
                        # if first_saved:
                        #     df.to_csv('first_face.csv',index = False)
                        #     first_saved = False

                        # 将文件名和 DataFrame 存储在列表中
                        data_frames.append((filename, df))

                        if verbose:
                            print(f"Loaded file: {filename} from {root} with session {session_name}")
    
        return data_frames
    
    def load_labels(self, data_dir, row_per_second = 100, challenge1 = 'union'):
        """
        加载标签到数据列表
        将时间段转换成时间戳的形式
        四个标签取法:robot,user,intersect,union,单独取两者,交集,并集
        """
        # ch1和ch2是两个不同的挑战
        dataframes_ch1_1 = []   # challenge1 robot的标签
        dataframes_ch1_2 = []   # challenge1 user的标签
        dataframes_ch1 = [] # 返回的challenge1标签,可以是交集合,并集或者单独的集合
        dataframes_ch2 = []

        # # 测试一下是否处理成功
        # first_ch1 = True
        # first_ch2 = True

        for challenge in ['challenge1_train', 'challenge2_train']:
            # challenge_path两个path的相对地址
            challenge_path = os.path.join(data_dir, challenge)
            if not os.path.exists(challenge_path):
                continue

            for filename in sorted(os.listdir(challenge_path), key=self.extract_file_number):
                # 只要.csv
                if not filename.endswith('.csv'):
                    continue
                frame_interval = 0.01
                # 处理 challenge1_train 数据
                if challenge == 'challenge1_train':

                    # 处理 robot 标签
                    if filename.startswith('challenge1_robot'):
                        df = pd.read_csv(os.path.join(challenge_path, filename))

                        if 'error_id' in df.columns:
                            df['challenge1'] = 1
                            df.drop(columns = ['error_id'], inplace = True)

                        df['error_onset'] = df['error_onset'].apply(self.time_str_to_seconds)
                        df['error_offset'] = df['error_offset'].apply(self.time_str_to_seconds)

                        df['error_onset_frame'] = (df['error_onset'] / frame_interval).astype(int)
                        df['error_offset_frame'] = (df['error_offset'] / frame_interval).astype(int)

                        new_rows = []
                        for index, row in df.iterrows():
                            for frame in np.arange(row['error_onset_frame'], row['error_offset_frame'] + 1):
                                new_row = row.copy()
                                new_row['frame'] = frame
                                new_rows.append(new_row)

                        expand_df = pd.DataFrame(new_rows)
                        expand_df.drop(columns=['error_onset', 'error_offset', 'error_onset_frame', 'error_offset_frame'], inplace=True)
                        expand_df = expand_df[[expand_df.columns[0], expand_df.columns[2], expand_df.columns[1]]]
                        dataframes_ch1_1.append((filename, expand_df))

                    # 处理 user 标签
                    elif filename.startswith('challenge1_user'):
                        df = pd.read_csv(os.path.join(challenge_path, filename))

                        if 'reaction_to' in df.columns:
                            df['challenge1'] = 1
                            df.drop(columns = ['reaction_to'], inplace = True)

                        df['reaction_onset'] = df['reaction_onset'].apply(self.time_str_to_seconds)
                        df['reaction_offset'] = df['reaction_offset'].apply(self.time_str_to_seconds)

                        df['reaction_onset_frame'] = (df['reaction_onset'] / frame_interval).astype(int)
                        df['reaction_offset_frame'] = (df['reaction_offset'] / frame_interval).astype(int)

                        new_rows = []
                        for index, row in df.iterrows():
                            for frame in np.arange(row['reaction_onset_frame'], row['reaction_offset_frame'] + 1):
                                new_row = row.copy()
                                new_row['frame'] = frame
                                new_rows.append(new_row)

                        expand_df = pd.DataFrame(new_rows)
                        expand_df.drop(columns=['reaction_onset', 'reaction_offset', 'reaction_onset_frame', 'reaction_offset_frame'], inplace=True)
                        expand_df = expand_df[[expand_df.columns[0], expand_df.columns[2], expand_df.columns[1]]]
                        dataframes_ch1_2.append((filename, expand_df))  # 注意：建议用另一个 list 保存 user 的数据

                elif challenge == 'challenge2_train':
                    df = pd.read_csv(os.path.join(challenge_path, filename))  # 必须先读文件！
                    # print(f"[DEBUG] 文件 {filename} 的列为：", df.columns.tolist())
                    if 'reaction_type' in df.columns:
                        df['challenge2'] = 1
                        df.drop(columns=['reaction_type'], inplace=True)
                    
                    # .apply是对一列数据循环使用这个函数
                    df['reaction_onset'] = df['reaction_onset'].apply(self.time_str_to_seconds)
                    df['reaction_offset'] = df['reaction_offset'].apply(self.time_str_to_seconds)

                    # 计算每行的帧号
                    df['reaction_onset_frame'] = (df['reaction_onset'] / frame_interval).astype(int)
                    df['reaction_offset_frame'] = (df['reaction_offset'] / frame_interval).astype(int)

                    new_rows = []

                    for index, row in df.iterrows():
                        start_frame = row['reaction_onset_frame']
                        end_frame = row['reaction_offset_frame']

                        # 生成从start->end的所有帧
                        frames = np.arange(start_frame, end_frame+1)

                        # 为每一帧添加行号，且保证之前的数据进行切片填充
                        for frame in frames:
                            new_row = row.copy()
                            new_row['frame'] = frame
                            new_rows.append(new_row)
                    # 添加新的列,删除原本的时间段这一些列
                    expand_df = pd.DataFrame(new_rows)
                    expand_df.drop(columns = ['reaction_onset', 'reaction_offset', 'reaction_onset_frame', 'reaction_offset_frame'], inplace=True)
                    expand_df = expand_df[[expand_df.columns[0], expand_df.columns[2], expand_df.columns[1]]]
                    
                    # # 测试
                    # if first_ch2:
                    #     expand_df.to_csv('first_label2.csv', index = False)
                    #     first_ch2 = False
                    dataframes_ch2.append((filename, expand_df))
            # 单独取robot
        if challenge1 == 'robot':
            dataframes_ch1 = dataframes_ch1_1
        # 单独取user
        elif challenge1 == 'user':
            dataframes_ch1 = dataframes_ch1_2  
        # 取交集或者并集
        elif challenge1 == 'intersect' or challenge1 == 'union':
            for filename_robot, df_robot in dataframes_ch1_1:
                match = re.search(r'labels_(.*?)_train', filename_robot)
                match = match.group(1)
                df_user = pd.DataFrame()
                for filename_user, df in dataframes_ch1_2:
                    match_2 = re.search(r'labels_(.*?)_train', filename_user)
                    match_2 = match_2.group(1)
                    if match == match_2:
                        df_user = df.copy()
                        break
                # 取交集
                if challenge1 == 'intersect':
                    intersection = pd.merge(df_robot, df_user,
                                            on = ['trial_name', 'frame'], how='inner')
                    intersection['challenge1'] = 1
                    dataframes_ch1.append((filename_robot, intersection))
                elif challenge1 == 'union':
                    union = pd.concat([df_robot, df_user], ignore_index=True)
                    # 按 trial_name 和 frame 去重，保留 challenge1 = 1
                    union = union.drop_duplicates(subset=['trial_name', 'frame'])
                    dataframes_ch1.append((filename_robot, union))

        return dataframes_ch1, dataframes_ch2


    def merge_X_data(self, face_head, audio, transcript, filename):
        """
        合并两个模态的特征: 脸部和头部,音频
        需要注意survival场景下,8号仅有图像特征,这个需要自动过滤掉
        """


        # 取filename以'-'为界的前两部分为匹配关系
        base_name = '-'.join(filename.split('-')[:2])
    
        match_face_df = None
        match_audio_df = None
        match_transcript_df = None

        # 匹配出三个模态相应session,一样的才能进行组合
        for face_name, face_df in face_head:
            if base_name in face_name:
                match_face_df = face_df
                break
                
        for audio_name, audio_df in audio:
            if base_name in audio_name:
                match_audio_df = audio_df
                break

        for transcript_name, transcript_df in transcript:
            if base_name in transcript_name:
                match_transcript_df = transcript_df
                break
        
        # 保留两个共有列
        face_static_cols = ['trial_name', 'frame']
        audio_static_cols = ['trial_name', 'frame']
        
        # 对另外的特征增加后缀,_face, _audio    
        face_features_cols = [col for col in match_face_df.columns if col not in face_static_cols]
        audio_features_cols = [col for col in match_audio_df.columns if col not in audio_static_cols]
        
        match_face_df = match_face_df.copy()
        match_audio_df = match_audio_df.copy()

        match_face_df.rename(columns = {col: col + '_face' for col in face_features_cols}, inplace = True)
        match_audio_df.rename(columns = {col: col + '_audio' for col in audio_features_cols}, inplace = True)


        # 合并 how = inner 是交集合并, how = outer 是并集合并
        # 使用how=inner表明二者取这两列的公有部分
        merged_df = pd.merge(match_face_df, match_audio_df, on = ['trial_name', 'frame'], how = 'inner')

        return merged_df, match_transcript_df
    
    # 81个场景8-折交叉验证划分
    def load_fold_info(self, merge_df):
        """
        将81个trial_name进行8折交叉验证划分。
        每个key是1~8,对应的value是该折的验证集trial_name列表
        前7折每折10个验证集,最后一折11个验证集
        使用随机种子保证每次划分结果一致。
        """
        trial_names = merge_df['trial_name'].unique().tolist()
        # 使用随机种子,保证每一次的划分是相同的
        random.seed(42)
        random.shuffle(trial_names)

        fold_info = {}
        fold_size = len(trial_names) // 8  # 每折大小
        for i in range(1, 9):  # 折编号从1到8
            start_idx = (i - 1) * fold_size
            end_idx = i * fold_size if i < 8 else len(trial_names)
            fold_info[i] = trial_names[start_idx:end_idx]

        print('\n8折交叉验证的验证集: ', fold_info)
        return fold_info


    def add_0_into_label(self, label1, label2, merge_df):
        """
        补充label缺失的帧，补充0，保证label和merge_df的场景和帧一致。
        
        Args:
            label1: 原始标签1数据，只包含 challenge1 == 1 的帧
            label2: 原始标签2数据，只包含 challenge2 == 1 的帧
            merge_df: 完整的feature帧数据，包含所有 trial_name 和 frame
        
        Returns:
            完整补齐后的 label1 和 label2，challenge1 和 challenge2 在原有基础上补 0
        """
        # 提取唯一索引（必须确保 merge_df 本身没有重复）
        full_index = merge_df[['trial_name', 'frame']].drop_duplicates()
        print('full_index长度:', len(full_index))

        # --- 关键修复：合并前对 label1/label2 去重 ---
        # 保留每个 (trial_name, frame) 最后一次出现的值（避免重复导致合并膨胀）
        label1 = label1.drop_duplicates(subset=['trial_name', 'frame'], keep='last')
        label2 = label2.drop_duplicates(subset=['trial_name', 'frame'], keep='last')

        # 左连接合并，缺失值补0
        merged1 = pd.merge(
            full_index,
            label1[['trial_name', 'frame', 'challenge1']],  # 只保留必要列
            on=['trial_name', 'frame'],
            how='left'
        ).fillna({'challenge1': 0}).astype({'challenge1': int})

        merged2 = pd.merge(
            full_index,
            label2[['trial_name', 'frame', 'challenge2']],
            on=['trial_name', 'frame'],
            how='left'
        ).fillna({'challenge2': 0}).astype({'challenge2': int})

               # 检查合并后的数据长度是否一致
        if len(merged1) != len(merged2):
            print("合并后的 label1 和 label2 的长度不一致！")
        # 检查 `1` 的数量是否发生变化
        original_ones_label1 = label1['challenge1'].sum()
        original_ones_label2 = label2['challenge2'].sum()
        new_ones_label1 = merged1['challenge1'].sum()
        new_ones_label2 = merged2['challenge2'].sum()

        print(f"Label1 中原有 1 的数量: {original_ones_label1}, 处理后 Label1 中 1 的数量: {new_ones_label1}")
        print(f"Label2 中原有 1 的数量: {original_ones_label2}, 处理后 Label2 中 1 的数量: {new_ones_label2}")

        return merged1, merged2
    

    def check_missing_frames(self, label1, label2, merge_df):
        """
        检查 label1 和 label2 中存在，但在 merge_df 中缺失的帧

        Args:
            label1: 原始标签1数据
            label2: 原始标签2数据
            merge_df: 完整的 feature 帧数据
        
        Returns:
            missing_label1: 在 label1 中有但在 merge_df 中没有的帧
            missing_label2: 在 label2 中有但在 merge_df 中没有的帧
        """
        # 提取 label1 和 label2 中的 trial_name 和 frame 列
        label1_frames = label1[['trial_name', 'frame']].drop_duplicates()
        label2_frames = label2[['trial_name', 'frame']].drop_duplicates()
        merge_df_frames = merge_df[['trial_name', 'frame']].drop_duplicates()

        # 找出 label1 和 label2 中有而 merge_df 中没有的帧
        missing_label1 = pd.merge(label1_frames, merge_df_frames, on=['trial_name', 'frame'], how='left', indicator=True).query('_merge == "left_only"').drop(columns=['_merge'])
        missing_label2 = pd.merge(label2_frames, merge_df_frames, on=['trial_name', 'frame'], how='left', indicator=True).query('_merge == "left_only"').drop(columns=['_merge'])

        # 输出缺失的帧
        print("Label1中有但feature中缺失的帧：")
        print(len(missing_label1))

        print("Label2中有但feature中缺失的帧：")
        print(len(missing_label2))
        return missing_label1, missing_label2

    # 用于非深度学习传统风格模型-> format = "classic"
    def get_summary_format(self, interval_length: int, stride_train: int, stride_eval: int, fps: int = 100, label_creation: str = "full", summary: str = 'mean', oversampling_rate: float = 0, undersampling_rate: float = 0, task: int=2, fold: int = 8, rescaling: str = None, start_padding: bool = False,challenge=2) -> tuple:
        """Convert the data to summary form. Split the data from the dfs into intervals of length interval_length with stride stride. Split takes place of adjacent frames of the same session.

        Args:
            interval_length: The length of the intervals
            stride_train: The stride for the training data (oversampling technique)
            stride_eval: The stride for the evaluation data (eval update frequency)
            fps: The desired fps of the data. Original is 100 fps
            label_creation: Either 'full' or 'stride_eval' or 'stride_train'. If 'full' the labels are based on mean of the whole interval, if 'stride' the labels are based on the mean of the stride. This does not affect the final eval but just the optimization goal during training.
            summary: The summary type. One of 'mean', 'max', 'min', 'median'
            oversampling_rate: x% of the minority class replicated in the training data as oversampling
            undersampling_rate: x% of the majority class removed from the training data as undersampling
            task: The task to load the data for. 1 for UserAwkwardness, 2 for RobotMistake, 3 for InteractionRupture
            fold: Fold which the validation data belongs to
            rescaling: The rescaling method. One of 'standardization', 'normalization', "none"

        Returns:
            The data in summary format

        Raises:
            ValueError: If the summary is not one of 'mean', 'max', 'min', 'median
        """
        #[10*[per_length,48.interval_length]];[10*[per_length,]]; [total_length,48.interval_length];[total_length,];[48,]
        val_X_TS, val_Y_summary_list, train_X_TS, train_Y_summary, column_order = self.get_timeseries_format(
            interval_length=interval_length, stride_train=stride_train, stride_eval=stride_eval, fps=fps, verbose=True,label_creation=label_creation, 
            oversampling_rate=oversampling_rate, undersampling_rate=undersampling_rate, task=task, fold=fold, rescaling=rescaling, start_padding=start_padding,challenge=challenge)
        
        if summary not in ['mean', 'max', 'min', 'median']:
            raise ValueError("Summary must be one of 'mean', 'max', 'min', 'median'")
        elif summary == 'mean':
            train_X_summary = np.mean(train_X_TS, axis=2)
            val_X_summary_list = [np.mean(val_X_TS[i], axis=2) for i in range(len(val_X_TS))]
        elif summary == 'max':
            train_X_summary = np.max(train_X_TS, axis=2)
            val_X_summary_list = [np.max(val_X_TS[i], axis=2) for i in range(len(val_X_TS))]
        elif summary == 'min':
            train_X_summary = np.min(train_X_TS, axis=2)
            val_X_summary_list = [np.min(val_X_TS[i], axis=2) for i in range(len(val_X_TS))]
        elif summary == 'median':
            train_X_summary = np.median(train_X_TS, axis=2)
            val_X_summary_list = [np.median(val_X_TS[i], axis=2) for i in range(len(val_X_TS))]
        
        train_X_summary=np.nan_to_num(train_X_summary)
        for i in range(len(val_X_summary_list)):
            val_X_summary_list[i] = np.nan_to_num(val_X_summary_list[i])
        return val_X_summary_list, val_Y_summary_list, train_X_summary, train_Y_summary, column_order
     # 重采样，将数据从100fps->50,20,10,5,2fps向下进行采样 ✅
    def resample(self, interval: list, fps: int, style: str) -> list:
        """Resample the interval to the desired fps. Original framerate is 100 fps.

        Args:
            interval: The interval to downsample
            fps: The desired fps
            style: The style of resampling. One of 'mean', 'max', 'min'

        Returns:
            The downsampled interval

        Raises:
            ValueError: If the style is not one of 'mean', 'max', 'min'
        """
        # Validate style
        if style not in ['mean', 'max', 'min']:
            raise ValueError("Style must be one of 'mean', 'max', 'min'")
        step = int(100 / fps)
        new_interval = []
        # Iterate over each feature in the interval
        for feature in interval:
            # Convert feature to a NumPy array for vectorized operations
            feature = np.array(feature)
            # Determine the shape of the new downsampled feature
            new_length = len(feature) // step
            reshaped_feature = feature[:new_length * step].reshape(-1, step)
            # Apply the selected downsampling style
            if style == 'mean':
                new_feature = np.mean(reshaped_feature, axis=1)
            elif style == 'max':
                new_feature = np.max(reshaped_feature, axis=1)
            elif style == 'min':
                new_feature = np.min(reshaped_feature, axis=1)
            # Append the downsampled feature to new_interval
            new_interval.append(new_feature.tolist())

        return new_interval

    # 用于时间序列的深度学习模型-> formate = "timeseries"
    def get_timeseries_format(self, interval_length: int, stride_train: int, stride_eval: int, fps: int = 100, 
                         verbose: bool = False, label_creation: str = "full", oversampling_rate: float = 0, 
                         undersampling_rate: float = 0, task: int=2, fold: int = 8, rescaling=None, 
                         start_padding: bool = False ) -> tuple:
        """Convert the data to timeseries form. Split the data from the dfs into intervals of length interval_length with stride stride.
        
        [保持原始文档字符串不变]
        """
        # 参数验证
        if rescaling not in ['standardization', 'normalization', 'none']:
            raise ValueError("Rescaling must be one of 'standardization', 'normalization', 'none'")
        if label_creation not in ['full', 'stride_eval', 'stride_train']:
            raise ValueError("label_creation must be one of 'full', 'stride_eval, 'stride_train'")
        
        # 根据challenge选择要处理的标签
        if task == 1:
            label_col = 'challenge1'
            all_Y = self.all_Y1[0]
            # val_Y = self.val_Y1
        else:  # challenge == 2
            label_col = 'challenge2'
            all_Y = self.all_Y2[0]
            # val_Y = self.all_Y2[0]
        
        # 获取训练和验证会话
        if fold not in self.fold_info:
            print("Training on all data, no validation")
            val_sessions = []
        else:
            val_sessions = self.fold_info[fold]
        
        train_sessions = []
        for f in self.fold_info:
            if f != fold:
                train_sessions.extend(self.fold_info[f])
        
        # 根据会话重新定义训练和验证数据
        self.train_X = self.all_X[0][0][self.all_X[0][0]['trial_name'].isin(train_sessions)]
        self.val_X = self.all_X[0][0][self.all_X[0][0]['trial_name'].isin(val_sessions)]
        # train_Y = all_Y[all_Y['trial_name'].isin(train_sessions)]
        # val_Y = all_Y[all_Y['trial_name'].isin(val_sessions)]
        # 报错修改,外部无法访问到val_Y
        self.train_Y = all_Y[all_Y['trial_name'].isin(train_sessions)]
        self.val_Y = all_Y[all_Y['trial_name'].isin(val_sessions)]
        
        if verbose:
            print(f"Train sessions: {len(train_sessions)}")
            print(f"\nVal sessions fold {fold}: {len(val_sessions)}")
            print(self.train_X["trial_name"].unique())
            print(self.val_X["trial_name"].unique())
        
        # 初始化结果容器
        val_Y_TS_list = []
        val_X_TS_list = []
        train_Y_TS = []
        train_X_TS = []
        
        # 处理训练数据
        for session in self.train_X['trial_name'].unique():
            # 提取当前session的数据
            session_df = self.train_X[self.train_X['trial_name'] == session]
            # 作训练丢掉无关列,前提是保证feature和label的frame完全对齐的
            session_df = session_df.drop(columns=['trial_name','frame','confidence_face'])
            column_order = session_df.columns
            
            # 裁剪多余帧
            cut_length = len(session_df) % interval_length
            if cut_length > 0:
                session_df = session_df[:-cut_length]
            
            # 标准化/归一化
            if rescaling == 'standardization':
                session_df = (session_df - session_df.mean()) / session_df.std()
            elif rescaling == 'normalization':
                session_df = (session_df - session_df.min()) / (session_df.max() - session_df.min())
            
            # 获取标签
            session_labels = self.train_Y[self.train_Y['trial_name'] == session]
            # session_labels=session_labels[:-cut_length]
            
            # 处理填充
            if start_padding:
                # 特征填充
                padding = np.zeros((interval_length-stride_train, session_df.shape[1]))
                session_df = pd.concat([pd.DataFrame(padding, columns=session_df.columns), session_df])
                
                # 标签填充
                padding = np.zeros((interval_length-stride_train, session_labels.shape[1]), dtype=int)
                session_labels = pd.concat([pd.DataFrame(padding, columns=session_labels.columns), session_labels])
            
            # 滑动窗口处理
            for i in range(0, len(session_df), stride_train):
                # 确保剩余数据足够构成一个完整窗口
                if i + interval_length > len(session_df):
                    break
                # 提取当前窗口的特征，并转置为(特征数, 时间步长) shape:[48,interval_length]
                interval = session_df.iloc[i:i+interval_length].values.T
                
                # 降采样
                if fps < 100:
                    interval = self.resample(interval, fps=fps, style='mean')
                    
                # 提取标签并处理 shape:[interval_length,1]
                labels = session_labels.iloc[i:i+interval_length][label_col].values.T
                
                # 根据标签创建策略选择标签 通过投票众数来将帧级别标签转换成窗口级别标签
                if label_creation == 'full':
                    final_label = np.argmax(np.bincount(labels))
                elif label_creation == 'stride_eval':
                    final_label = np.argmax(np.bincount(labels[-stride_eval:]))
                elif label_creation == 'stride_train':
                    final_label = np.argmax(np.bincount(labels[-stride_train:]))
                
                # 添加数据和标签
                train_X_TS.append(interval)
                train_Y_TS.append(final_label)
        
        # 处理验证数据
        for session in self.val_X['trial_name'].unique():
            val_X_TS = []
            val_Y_TS = []
            
            session_df = self.val_X[self.val_X['trial_name'] == session]
            session_df = session_df.drop(columns=['trial_name','frame','confidence_face'])
            
            cut_length = len(session_df) % interval_length
            if cut_length > 0:
                session_df = session_df[:-cut_length]
                
            if rescaling == 'standardization':
                session_df = (session_df - session_df.mean()) / session_df.std()
            elif rescaling == 'normalization':
                session_df = (session_df - session_df.min()) / (session_df.max() - session_df.min())
            
            # session_labels和 session_df的长度不一样，session_labels是原始的，没有考虑cut_length
            session_labels = self.val_Y[self.val_Y['trial_name'] == session]
            # session_labels=session_labels[:-cut_length]
            
            if start_padding:
                padding = np.zeros((interval_length-stride_eval, session_df.shape[1]))
                session_df = pd.concat([pd.DataFrame(padding, columns=session_df.columns), session_df])
                
                padding = np.zeros((interval_length-stride_eval, session_labels.shape[1]), dtype=int)
                session_labels = pd.concat([pd.DataFrame(padding, columns=session_labels.columns), session_labels])
            
            for i in range(0, len(session_df), stride_eval):
                if i + interval_length > len(session_df):
                    break
                
                #[48,interval_length]
                interval = session_df.iloc[i:i+interval_length].values.T
                
                if fps < 100:
                    interval = self.resample(interval, fps=fps, style='mean')
                    
                # 提取标签 - 验证集始终使用整个窗口的众数
                # 这里的labels是一个一维数组，shape:[interval_length,1]
                labels = session_labels.iloc[i:i+interval_length][label_col].values.T
                final_label = np.argmax(np.bincount(labels))
                
                val_X_TS.append(interval)
                val_Y_TS.append(final_label)
            
            val_X_TS_list.append(val_X_TS)
            val_Y_TS_list.append(val_Y_TS)
        
        # 转换为numpy数组
        
        train_X_TS = np.array(train_X_TS)#[length,48,interval_length]
        train_Y_TS = np.array(train_Y_TS)#[length,]
        
        for i in range(len(val_X_TS_list)):
            val_X_TS_list[i] = np.array(val_X_TS_list[i])
            val_Y_TS_list[i] = np.array(val_Y_TS_list[i])
        
        # 获取当前challenge的少数类和多数类
        minority_class = np.argmin(np.bincount(train_Y_TS))
        majority_class = np.argmax(np.bincount(train_Y_TS))
        
        if verbose:
            print(f"Minority class: {minority_class}")
            print(f"Majority class: {majority_class}")
        
        # 处理过采样
        if oversampling_rate > 0:
            minority_indices = np.where(train_Y_TS == minority_class)[0]
            oversampling_indices = np.random.choice(
                minority_indices, 
                int(len(minority_indices) * oversampling_rate),
                replace=True
            )
            train_X_TS = np.concatenate((train_X_TS, train_X_TS[oversampling_indices]), axis=0)
            train_Y_TS = np.concatenate((train_Y_TS, train_Y_TS[oversampling_indices]), axis=0)
            
            if verbose:
                print(f"From minority class: {len(minority_indices)}, oversampled: {len(oversampling_indices)}")
        
        # 处理欠采样
        if undersampling_rate > 0:
            majority_indices = np.where(train_Y_TS == majority_class)[0]
            undersampling_indices = np.random.choice(
                majority_indices,
                int(len(majority_indices) * undersampling_rate),
                replace=False
            )
            train_X_TS = np.delete(train_X_TS, undersampling_indices, axis=0)
            train_Y_TS = np.delete(train_Y_TS, undersampling_indices, axis=0)
            
            if verbose:
                print(f"From majority class: {len(majority_indices)}, undersampled: {len(undersampling_indices)}")
        
        
        return val_X_TS_list,val_Y_TS_list, train_X_TS, train_Y_TS, column_order
#     def get_timeseries_format_test_data(self, interval_length: int, stride_eval: int, fps: int = 100, verbose: bool = False, label_creation: str = "full", task: int = 2, rescaling: str = "none", start_padding: bool = False) -> tuple:
#         """ Convert the data to timeseries form. Split the data from the dfs into intervals of length interval_length with stride stride. Split takes place of adjacent frames of the same session.

#         Args:
#             interval_length: The length of the intervals
#             stride_train: The stride for the training data (oversampling technique)
#             stride_eval: The stride for the evaluation data (eval update frequency)
#             fps: The desired fps of the data. Original is 100 fps
#             verbose: Print debug information
#             label_creation: Either 'full' or 'stride_eval' or 'stride_train'. If 'full' the labels are based on mean of the whole interval, if 'stride' the labels are based on the mean of the stride. This does not affect the final eval but just the optimization goal during training.
#             rescaling: The rescaling method. One of 'standardization', 'normalization', 'none'
#             start_padding: If True, the data is padded at the start with 0s, and the actual data starting for the last stride elements

#         Returns:
#             The data in timeseries format and the column order for feature importance analysis

#         Raises:
#             ValueError: If the label_creation is not one of 'full', 'stride_eval', 'stride_train'
#             ValueError: If the rescaling is not one of 'standardization', 'normalization', 'none
#         """

# # # 验证取值
# data_loader = DataLoader_HRI2()
# interval_length = 50  # 时间窗口长度
# stride_train = 10     # 训练数据滑动步长
# stride_eval = 20      # 评估数据滑动步长
# summary = 'mean'      # 统计方法：平均值
# fold = 1              # 使用第1折进行验证
# challenge = 2        # 选择挑战2

# val_X_summary_list, val_Y_summary_list, train_X_summary, train_Y_summary, column_order = data_loader.get_summary_format(
#     interval_length=interval_length,
#     stride_train=stride_train,
#     stride_eval=stride_eval,
#     fps=100,
#     label_creation='full',
#     summary=summary,
#     oversampling_rate=0.5,  # 过采样率
#     undersampling_rate=0,   # 欠采样率
#     task=2,
#     fold=fold,
#     rescaling='standardization',
#     start_padding=False
# )

# # 打印结果形状信息
# print(f"验证集特征形状: {[x.shape for x in val_X_summary_list]}")
# print(f"验证集标签形状: {[len(x) for x in val_Y_summary_list]}")
# print(f"训练集特征形状: {train_X_summary.shape}")
# print(f"训练集标签形状: {train_Y_summary.shape}")
# print(f"特征数量: {len(column_order)}")

# # 检查特征和标签的前几个样本
# print("\n训练集特征前3个样本，前5个特征:")
# print(train_X_summary[:3, :5])

# print("\n训练集标签前10个样本:")
# print(train_Y_summary[:10])