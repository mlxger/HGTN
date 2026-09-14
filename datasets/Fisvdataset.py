from torch.utils.data import Dataset
import numpy as np
import os
import torch
import torch.nn.functional as F


class FisvDataset(Dataset):
    def __init__(self, video_feat_path, label_path, clip_num=124, action_type='all',
                 score_type='Total_Score', train=True):
        self.train = train
        self.video_path = video_feat_path
        self.erase_path = video_feat_path + '_erTrue'
        if score_type == 'TES':
            self.score_range = 45
        elif score_type == 'PCS':
            self.score_range = 40

        self.clip_num = clip_num
        self.labels = self.read_label(label_path, score_type, action_type)

    def read_label(self, label_path, score_type, action_type):
        fr = open(label_path, 'r')
        idx = {'TES': 1, 'PCS': 2, 'Total_Score': 3}
        labels = []
        score = []
        for i, line in enumerate(fr):
            # if i == 0:
            #     continue
            line = line.strip().split()               #  ['1', '28.71', '29.36', '1']
            s = float(line[idx[score_type]])
            labels.append([line[0], s])
            score.append(s)

        print('labels:',labels)
        print("max:",max(score))
        print("min:",min(score))
        return labels

    def __getitem__(self, idx):
        video_feat = np.load(os.path.join(self.video_path, self.labels[idx][0] + '.npy'))

        if len(video_feat) > self.clip_num:
            st = np.random.randint(0, len(video_feat) - self.clip_num)
            video_feat = video_feat[st:st + self.clip_num]
            # erase_feat = erase_feat[st:st + self.clip_num]
        elif len(video_feat) < self.clip_num:
            new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
            new_feat[:video_feat.shape[0]] = video_feat
            video_feat = new_feat


        video_feat = torch.from_numpy(video_feat).float()
        #print(video_feat.shape)
        return video_feat, self.normalize_score(self.labels[idx][1])

    def __len__(self):
        return len(self.labels)

    def normalize_score(self, score):
        return score / self.score_range

    # def normalize_score(self, score):
    #     Difficulty_Score TES
    #     return score / 45

    # difficulty_score is TES
    # execution_score is PCS






