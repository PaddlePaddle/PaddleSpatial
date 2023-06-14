import glob
import datetime
import pandas as pd
import argparse
import random
import os
random.seed(0)

month_days = [31,28,31,30,31,30,31,31,30,31,30,31]
time_zone_dict = {}
for i in range(3, 9):
    time_zone_dict[str(i).zfill(2)] = '0'
for i in range(9, 15):
    time_zone_dict[str(i).zfill(2)] = '1'
for i in range(15, 21):
    time_zone_dict[str(i).zfill(2)] = '2'
for i in list(range(21, 24)) + list(range(0, 3)):
    time_zone_dict[str(i).zfill(2)] = '3'
    
def now_time_zone(times):
    md, h = times.split()
    h = h[:2]
    m, d = md.split('/')[:2]
    zone = time_zone_dict[h]
    if h in ['00','01','02']:
        if int(d) == 1:
            m = str(int(m)-1) if m != '1' else '12'
            d = str(month_days[int(m)-1])
        else:
            d = str(int(d)-1)
    return int(zone), m + '#' + d

def load_mobility_dict_with_time(trips, stations):
    stat_dict = {}
    for index, row in stations.iterrows():
        stat_dict[row['id']] = {'name': row['name'], 'loc': (float(row['latitude']), float(row['longitude']))}
        
    flow_dict = {}
    cnt = 0
    for index, row in trips.iterrows():
        start_id, end_id = row['from_station_id'], row['to_station_id']
        start_time, end_time = row['start_time'], row['end_time']
        start_zone, _ = now_time_zone(start_time)
        end_zone, _ = now_time_zone(end_time)
        if start_zone != end_zone:
            continue
        cnt += 1 
        delta = (datetime.datetime.strptime(end_time,"%m/%d/%Y %H:%M:%S")-datetime.datetime.strptime(start_time,"%m/%d/%Y %H:%M:%S")).seconds/60
        ids = (start_id, end_id)
        if start_id not in flow_dict:
            flow_dict[start_id] = {}
        if end_id not in flow_dict[start_id]:
            flow_dict[start_id][end_id] = [[] for _ in range(4)]     
        flow_dict[start_id][end_id][start_zone].append(delta)
    
    flow_sorted_list = [[] for _ in range(4)]
    for k1, v1 in flow_dict.items():
        for k2, v2 in v1.items():
            for i, f in enumerate(v2):
                if len(f) > 1:
                    flow_sorted_list[i].append(len(f))      
    for i in range(4):
        flow_sorted_list[i] = sorted(flow_sorted_list[i])
    return flow_sorted_list, flow_dict, stat_dict

def build_relationship(flow_sorted_list, flow_dict, stat_dict):
    cnt_dict = {x: 0 for x in flow_dict}
    for k1, v1 in flow_dict.items():
        cnt_dict[k1] = sum([sum([len(x) for x in v2]) for v2 in v1.values()])
    threshold = 5
    relation = [[] for i in range(4)]
    relation_set = [set() for i in range(4)]
    for k1, v1 in flow_dict.items():
        for k2, v2 in v1.items():
            if k1 == k2:
                continue
            if k2 not in flow_dict or k1 not in flow_dict[k2]:
                continue
            for i in range(4):
                k1v = len(flow_dict[k2][k1][i])
                threshold = flow_sorted_list[i][int(len(flow_sorted_list[i])/2.0*1)]
                if len(v2[i]) > threshold and len(flow_dict[k2][k1][i]) > threshold and 1.0*len(v2[i])/(cnt_dict[k1]) > 0.0005 and 1.0*k1v/(cnt_dict[k2]) > 0.0005:
                    relation[i] += [(str(k1), str(k2), len(v2[i]))]
                    if k1 < k2:
                        relation_set[i].add((k1, k2))
                    else:
                        relation_set[i].add((k2, k1))
    # processing location nodes
    node_set = set()
    for rr in relation:
        for u, v, n in rr:
            if n < threshold: continue
            node_set.add(u)
            node_set.add(v)
    node_dict = {k: stat_dict[int(k)] for k in node_set}
    all_pair_set_dict = {'time'+str(i+1): set() for i in range(4)}
    for i in range(4):
        for u, v, n in relation[i]:
            uv = v+'#'+u if u > v else u+'#'+v
            all_pair_set_dict['time'+str(i+1)].add(uv)
    for i in range(4):
        for j in range(i+1, 4):
            s1 = all_pair_set_dict['time'+str(i+1)]
            s2 = all_pair_set_dict['time'+str(j+1)]
    all_pair_set = set()
    all_pair_list_d = {}
    for k, v in all_pair_set_dict.items():
        all_pair_set = all_pair_set | v
        all_pair_list_d[k] = list(v)
    return node_dict, relation, all_pair_list_d

def build_dataset(node_dict, relation, all_pair_list_d):
    all_pair_list_d['time0'] = [x for x in all_pair_list_d['time4']]
    ratio = 0.2
    test_val_pair_dict = {'time'+str(i+1): set() for i in range(4)}
    relation_g = dict()
    value_dict = dict()
    for i in range(4):
        for u, v, n in relation[i]:
            value_dict[u+'#'+v] = n
            value_dict[v+'#'+u] = n
            if v not in relation_g:
                relation_g[v] = {'time1': {}, 'time2': {}, 'time3': {}, 'time4': {}}
            relation_g[v]['time'+str(i+1)][u] = n
    relation_g_skip = dict()
    for k1, v in relation_g.items():
        relation_g_skip[k1] = {}
        for t, v2 in v.items():
            if len(v2) == 0:
                relation_g_skip[k1][t] = ''
                continue
            k2_list = [k2 for k2 in v2]
            ind = random.randint(0,len(k2_list)-1)
            k2 = k2_list[ind]
            relation_g_skip[k1][t] = k2
    for i in range(4):
        i = str(i+1)
        lll = all_pair_list_d['time'+i]
        num_test = int(len(lll) * ratio)
        while len(test_val_pair_dict['time'+i]) < num_test:
            ind = random.randint(0,len(lll)-1)
            if lll[ind] in test_val_pair_dict['time'+i]:
                continue
            b1, b2 = lll[ind].split('#')
            conflict = False
            for j in range(int(i), 5):
                if len(relation_g[b1]['time'+str(j)]) <= 1 or len(relation_g[b2]['time'+str(j)]) <= 1:
                    conflict = True
                    break
            if conflict: continue
            for j in range(int(i), 5):
                if b2 not in relation_g[b1]['time'+str(j)]:
                    continue
                test_val_pair_dict['time'+str(j)].add(lll[ind])
                relation_g[b1]['time'+str(j)].pop(b2)
                relation_g[b2]['time'+str(j)].pop(b1)

    train_pair_dict, test_pair_dict, valid_pair_dict = {'time0':[]}, {'time0':[]}, {'time0':[]}
    for i in range(1, 5):
        time = 'time'+str(i)
        ll = list(test_val_pair_dict[time])
        random.shuffle(ll)
        times = ['morning','midday','night','midnight']
        test_pair_dict[time] = [x+'#'+times[i-1] for x in ll[:int(len(ll)/2)]]
        valid_pair_dict[time] = [x+'#'+times[i-1] for x in ll[int(len(ll)/2):]]
        train_pair_dict[time] = [x+'#'+times[i-1] for x in list(set(all_pair_list_d[time])-set(ll))]
        test_pair_dict['time0'] += test_pair_dict[time]
        train_pair_dict['time0'] += train_pair_dict[time]
        valid_pair_dict['time0'] += valid_pair_dict[time]
    return train_pair_dict, valid_pair_dict, test_pair_dict, value_dict

def save_relationship_dataset(output_path, node_dict, value_dict, mobi_dict, train_pair_dict, valid_pair_dict, test_pair_dict):
    times = ['morning','midday','night','midnight']
    with open('%s/entities.dict' % output_path, 'w') as f:
        for i, (bid, value) in enumerate(node_dict.items()):
            f.write(str(i) + '\t' + bid + '\n')
    with open('%s/coords.dict' % output_path, 'w') as f:
        for i, (bid, value) in enumerate(node_dict.items()):
            f.write(bid + '\t' + str(value['coord'][0]) + '\t' + str(value['coord'][1]) + '\n')

    relation_types = ['high-flow', 'low-flow']
    idx = 0
    with open('%s/relations.dict' % output_path, 'w') as f:
        for r in relation_types:
            for t in times:
                f.write(str(idx) + '\t' + '_'.join([r, t]) + '\n')
                idx += 1

    f1 = open('%s/train.txt' % output_path, 'w')
    f2 = open('%s/valid.txt' % output_path, 'w')
    f3 = open('%s/test.txt' % output_path, 'w')
    cnt1, cnt2, cnt3 = 0, 0, 0
    for uv in train_pair_dict['time0']:
        u, v, t = uv.split('#')
        ti = times.index(t)
        rel_threshold = mobi_dict[ti][int(len(mobi_dict[ti])/4.0*3)]
        high_rel = int(value_dict[u+'#'+v] > rel_threshold)
        rel = relation_types[high_rel]
        f1.write(u + '\t' + '_at_'.join([rel, t]) + '\t' + v + '\n')
        high_rel = int(value_dict[v+'#'+u] > rel_threshold)
        rel = relation_types[high_rel]
        f1.write(v + '\t' + '_at_'.join([rel, t]) + '\t' + u + '\n')
    for uv in valid_pair_dict['time0']:
        u, v, t = uv.split('#')
        ti = times.index(t)
        rel_threshold = mobi_dict[ti][int(len(mobi_dict[ti])/4.0*3)]
        high_rel = int(value_dict[u+'#'+v] > rel_threshold)
        rel = relation_types[high_rel]
        f2.write(u + '\t' + '_at_'.join([rel, t]) + '\t' + v + '\n')
        high_rel = int(value_dict[v+'#'+u] > rel_threshold)
        rel = relation_types[high_rel]
        f2.write(v + '\t' + '_at_'.join([rel, t]) + '\t' + u + '\n')
    for uv in test_pair_dict['time0']:
        u, v, t = uv.split('#')
        ti = times.index(t)
        rel_threshold = mobi_dict[ti][int(len(mobi_dict[ti])/4.0*3)]
        high_rel = int(value_dict[u+'#'+v] > rel_threshold)
        rel = relation_types[high_rel]
        f3.write(u + '\t' + '_at_'.join([rel, t]) + '\t' + v + '\n')
        high_rel = int(value_dict[v+'#'+u] > rel_threshold)
        rel = relation_types[high_rel]
        f3.write(v + '\t' + '_at_'.join([rel, t]) + '\t' + u + '\n')
    f1.close()
    f2.close()
    f3.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, default='./data/raw/DIVVY')
    parser.add_argument('--output_path', type=str, default='./data/chicago/')
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # load dataset
    year = '2017'
    path = os.getcwd()
    stat_csv_files = glob.glob(os.path.join(args.input_data, "*Stations_%s*.csv"%year))
    trip_csv_files = glob.glob(os.path.join(args.input_data, "*Trips_%s*.csv"%year))
    stations = pd.read_csv(stat_csv_files[0])
    trips =  pd.read_csv(trip_csv_files[0])
    for i in range(1, len(stat_csv_files)):
        stat = pd.read_csv(stat_csv_files[i])
        stations = pd.concat([stations,stat],ignore_index=True)
    for i in range(1, len(trip_csv_files)):
        trip = pd.read_csv(trip_csv_files[i])
        trips = pd.concat([trips,trip],ignore_index=True)

    mobi_sort_list, mobi_dict, stat_dict = load_mobility_dict_with_time(trips, stations)
    node_dict, relation, all_pair_list_d = build_relationship(mobi_sort_list, mobi_dict, stat_dict)
    train, valid, test, value_dict = build_dataset(node_dict, relation, all_pair_list_d)

    save_relationship_dataset(args.output_path, node_dict, value_dict, mobi_sort_list, train, valid, test)