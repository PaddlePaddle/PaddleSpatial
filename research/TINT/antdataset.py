'''
Author: jianglinlang
Date: 2021-12-29 17:53:16
LastEditTime: 2022-05-23 15:52:45
LastEditors: jianglinlang
Description: antdataset
FilePath: /jianglinlang/paddle/antdataset.py
'''
from datetime import time
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
from collections import namedtuple


class TripDataset():
    def __init__(self, padding=True):
        if padding:
            self.poi2idx = {'pad': 0}
            self.idx2poi = ['pad']
            self.n_poi = 1

            self.n_user = 1
            self.user2idx = {'pad': 0}
            self.idx2user = ['pad']

            self.n_tag = 1
            self.tag2idx = {'pad': 0}
            self.idx2tag = ['pad']

            self.coord_list = [[0., 0.]]
            self.poiID2tag = [0]

            self.poiID2duration = [0.]
            self.poiID2distime = [0]

            self.max_time = 0.
        else:
            print("Must padding")

    def read_poi_file(self, filename):
        df = pd.read_csv(filename, header=0, sep=";")
        for _, row in df.iterrows():
            poiID, poiname, lat, lon, tag = row.values
            self.poi2idx[poiname] = poiID
            self.idx2poi.append(poiname)
            self.coord_list.append([lat, lon])

            if tag not in self.tag2idx:
                self.tag2idx[tag] = self.n_tag
                self.idx2tag.append(tag)
                self.n_tag += 1
            self.poiID2tag.append(self.tag2idx[tag])
        self.n_poi = len(self.idx2poi)

    def map_time(self, refined_trips, defaut_dura=600, interval=15):
        temppoi2duration = defaultdict(list)
        for trip in refined_trips:
            for checkin in trip:
                poi, duration = checkin[1], checkin[2][-1]
                if duration == 0:
                    continue
                else:
                    temppoi2duration[poi].append(duration)
        for poiID in range(1, self.n_poi):
            dura_list = temppoi2duration[poiID]
            if len(dura_list) == 0:
                self.poiID2duration.append(defaut_dura)
            else:
                self.poiID2duration.append(np.mean(dura_list))
        self.poiID2duration = np.array(self.poiID2duration) / 60
        self.poiID2distime = np.array(self.poiID2duration)
        self.poiID2distime /= 60
        self.poiID2distime = ((self.poiID2distime + 1e-6) // interval).astype(np.int32)

    def read_trans(self, filename, meter_per_min=500):
        dis_matrix = np.zeros((self.n_poi, self.n_poi))
        dis_df = pd.read_csv(filename, sep=";", header=0)
        for _, row in dis_df.iterrows():
            dis_matrix[row['from'], row['to']] = row['cost']
        self.trsf_matrix = dis_matrix / meter_per_min

    def split(self, refined_trips, city, val_prop=0.1, test_prop=0.1):
        data_num = len(refined_trips)
        test_num = int(data_num * test_prop)
        val_num = int(data_num * val_prop)
        train_num = data_num - val_num - test_num
        refined_trips_sort = sorted(refined_trips, key=lambda x: x[0][2][0])
        if not os.path.isdir(city):
            os.makedirs(city)
        self.output(refined_trips_sort[:train_num], os.path.join(city, "trainset"))
        self.output(refined_trips_sort[train_num: train_num + val_num], os.path.join(city, "validset"))
        self.output(refined_trips_sort[train_num + val_num:], os.path.join(city, 'testset'))

        self.save_expert(refined_trips_sort[:train_num], os.path.join(city, "expert_traj.pt"))

    def output(self, trips, filename):
        with open(filename, "w") as f:
            for trip in trips:
                user = trip[0][0]
                pois = [checkin[1] for checkin in trip]
                len_trip = len(pois)
                time_budget = 0.
                for i in range(1, len_trip):
                    time_budget += self.trsf_matrix[pois[i - 1], pois[i]] + self.poiID2duration[pois[i]]
                self.max_time = max(self.max_time, time_budget)
                f.write(str([user] + pois + [time_budget]) + '\n')

    def save_expert(self, trips, filename):
        users = []
        cur_pois = []
        actions = []
        time_lefts = []
        expert_traj = []
        for trip in trips:
            user = trip[0][0]
            pois = [checkin[1] for checkin in trip]
            len_trip = len(pois)
            time_left = [self.trsf_matrix[pois[-2], pois[-1]] + self.poiID2duration[pois[-1]]]
            for i in range(len_trip - 2, 0, -1):
                now_left = time_left[0]
                time_left.insert(0, now_left + self.trsf_matrix[pois[i - 1], pois[i]] + self.poiID2duration[pois[i]])
            for i in range(0, len_trip - 1):
                expert_traj.append([user, pois[i], time_left[i], pois[i + 1]])

    def determine_max_time(self, interval):
        max_budget_time = int((self.max_time + 1e-6) // interval)
        max_poi_time = max(self.poiID2distime)
        self.n_time = max(max_budget_time, max_poi_time)
        print(f"interval = {interval} mins. n_time is {self.n_time}")


class ANTDataset(TripDataset):
    def __init__(self, padding=True):
        super().__init__(padding=padding)

    def split(self, refined_trips, city, val_prop=0.1, test_prop=0.1):
        data_num = len(refined_trips)
        test_num = int(data_num * test_prop)
        val_num = int(data_num * val_prop)
        train_num = data_num - val_num - test_num
        refined_trips_sort = sorted(refined_trips, key=lambda x: x[0][2][0])
        if not os.path.isdir(city):
            os.makedirs(city)
        self.output(refined_trips_sort[:train_num], os.path.join(city, "trainsetANT"))
        self.output(refined_trips_sort[train_num: train_num + val_num], os.path.join(city, "validsetANT"))
        self.output(refined_trips_sort[train_num + val_num:], os.path.join(city, 'testsetANT'))

    def output(self, trips, filename):
        idx_in_src = list(range(self.n_poi))
        record = []
        for trip in trips:
            user = trip[0][0]
            pois = [checkin[1] for checkin in trip]
            len_trip = len(pois)
            u = [user] * self.n_poi
            u_t = [user] * len_trip
            t_t = [self.poiID2tag[poi] for poi in pois]

            time_limit = 0.
            for i in range(1, len_trip):
                time_limit += self.trsf_matrix[pois[i - 1], pois[i]] + self.poiID2duration[pois[i]]
            record.append([u, idx_in_src, self.poiID2tag, u_t, pois, t_t, time_limit, pois])
        with open(filename, "wb") as f:
            pickle.dump(record, f)
