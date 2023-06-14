from collections import Counter
import pandas as pd
import argparse
import random
import os
random.seed(0)

month_days = {'Apr': '30', 'May': '31', 'Jun': '30', 'Jul': '31', 'Aug': '31', 'Sep': '30', 'Oct': '31', 'Nov': '30', 'Dec': '31', 'Jan': '31', 'Feb': '28'}
month_list = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb']
time_zone_dict = {}
for i in range(3, 9):
    time_zone_dict[str(i).zfill(2)] = '0'
for i in range(9, 15):
    time_zone_dict[str(i).zfill(2)] = '1'
for i in range(15, 21):
    time_zone_dict[str(i).zfill(2)] = '2'
for i in list(range(21, 24)) + list(range(0, 3)):
    time_zone_dict[str(i).zfill(2)] = '3'

def now_time_zone(times, offset=9):
    m, d, h = times
    d, h = int(d), int(h)
    if offset > 0:
        if h + offset > 23:
            if d + 1 <= int(month_days[m]):
                d = d + 1
            else:
                d = 1
                m = month_list[month_list.index(m)+1]
            h = (h + offset) % 24
        else:
            h = h + offset
    else:
        if h + offset < 0:
            if d > 1:
                d = d - 1
            else:
                m = month_list[month_list.index(m)-1]
                d = month_days[m]
            h = 24 + h + offset 
        else:
            h = h + offset
    d, h = str(d).zfill(2), str(h).zfill(2)
    zone = time_zone_dict[h]
    if h in ['00','01','02']:
        if d == '01':
            m = month_list[month_list.index(m)-1]
            d = month_days[m]
        else:
            d = str(int(d)-1).zfill(2)
    return int(zone), m + '#' + d
        
def load_user_visit(data):
    visit_dict = {}
    for index, row in data.iterrows():
        user = row['userId']
        tag = row['venueCategory']
        time = row['utcTimestamp']
        location = row['venueId']
        x, y = row['latitude'], row['longitude']
        if location not in visit_dict:
            visit_dict[location] = {'user':[], 'loc': [], 'time': []}
        visit_dict[location]['user'].append(user)
        visit_dict[location]['time'].append(time[4:13].split(' '))
        visit_dict[location]['loc'].append((x,y,tag))

def update_user_visit_with_time(visit_dict):
    visit_dict_new = {}
    for k, v in visit_dict.items():
        if len(set(v['user'])) >= 4:
            visit_dict_new[k] = [set() for _ in range(4)]
            for time, user in zip(v['time'], v['user']):
                zone, md = now_time_zone(time)
                visit_dict_new[k][zone].add(md + '#' + str(user))
    # print(len(visit_dict_new), len(visit_dict))
    return visit_dict_new

def build_relationship(visit_dict_new):
    # basic correlations
    relation = [[] for i in range(4)]
    for k1, v1 in visit_dict_new.items():
        for k2, v2 in visit_dict_new.items():
            if k1 == k2:
                continue
            for i in range(4):
                v12 = v1[i] & v2[i]
                if len(v12) > 2:
                    relation[i] += [(k1, k2, len(v12))]
    
    # processing location nodes
    node_set = set()
    for rr in relation:
        for u, v, n in rr:
            if n < 4: continue
            node_set.add(u)
            node_set.add(v)
    poi_dict = {}
    for k in node_set:
        poi_dict[k] = {}
        xy_tag = visit_dict[k]['loc']
        xy = list(set([(x,y) for x,y,t in xy_tag]))
        tag = list(set([t for x,y,t in xy_tag]))
        xy_count = Counter([(x,y) for x,y,t in xy_tag])
        tag_count = Counter([t for x,y,t in xy_tag])
        xy_count = sorted(xy_count.items(), key=lambda x:x[1], reverse=True)
        tag_count = sorted(tag_count.items(), key=lambda x:x[1], reverse=True)
        if len(xy_count) == 1:
            poi_dict[k]['coord'] = xy_count[0][0]
        else:
            poi_dict[k]['coord'] = xy_count[0][0]
            pass
        if len(tag_count) == 1:
            poi_dict[k]['tag'] = tag_count[0][0]
        else:
            poi_dict[k]['tag'] = tag_count[0][0]

    threshold = 4
    all_pair_set_dict = {'time'+str(i+1): set() for i in range(4)}
    for i in range(4):
        for u, v, n in relation[i]:
            if n < threshold:
                continue
            uv = v+'#'+u if u > v else u+'#'+v
            all_pair_set_dict['time'+str(i+1)].add(uv)
    for i in range(4):
        for j in range(i+1, 4):
            s1 = all_pair_set_dict['time'+str(i+1)]
            s2 = all_pair_set_dict['time'+str(j+1)]
            print(len(s1), len(s2), str(i+1)+'#'+str(j+1), 1.0*len(s1&s2)/len(s1|s2), 1.0*len(s1&s2)/min(len(s1),len(s2)))
    all_pair_set = set()
    all_pair_list_d = {}
    for k, v in all_pair_set_dict.items():
        all_pair_set = all_pair_set | v
        all_pair_list_d[k] = list(v)

    return poi_dict, relation, all_pair_list_d

def build_dataset(node_dict, relation, all_pair_list_d):
    all_pair_list_d['time0'] = [x for x in all_pair_list_d['time4']]
    ratio = 0.2
    test_val_pair_dict = {'time'+str(i+1): set() for i in range(4)}
    threshold = 4
    relation_g = dict()
    for i in range(4):
        for u, v, n in relation[i]:
            if n < threshold:
                continue
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
                if len(relation_g[b1]['time'+str(j)]) == 1 or len(relation_g[b2]['time'+str(j)]) == 1:
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
        
        times = ['morning','midday','night','minight']
        test_pair_dict[time] = [x+'#'+times[i-1] for x in ll[:int(len(ll)/2)]]
        valid_pair_dict[time] = [x+'#'+times[i-1] for x in ll[int(len(ll)/2):]]
        train_pair_dict[time] = [x+'#'+times[i-1] for x in list(set(all_pair_list_d[time])-set(ll))]
        test_pair_dict['time0'] += test_pair_dict[time]
        train_pair_dict['time0'] += train_pair_dict[time]
        valid_pair_dict['time0'] += valid_pair_dict[time]

    node_dict = dict()
    for rr in relation:
        for u, v, n in rr:
            if n < threshold: continue
            if u not in node_dict:
                node_dict[u] = node_dict[u]
            if v not in node_dict:
                node_dict[v] = node_dict[v]

    return node_dict, train_pair_dict, valid_pair_dict, test_pair_dict

def save_relationship_dataset(output_path, node_dict, train_pair_dict, valid_pair_dict, test_pair_dict):
    times = ['morning','midday','night','minight']
    with open('%s/entities.dict' % output_path, 'w') as f:
        for i, (bid, value) in enumerate(node_dict.items()):
            f.write(str(i) + '\t' + bid + '\n')
    with open('%s/coords.dict' % output_path, 'w') as f:
        for i, (bid, value) in enumerate(node_dict.items()):
            f.write(bid + '\t' + str(value['coord'][0]) + '\t' + str(value['coord'][1]) + '\n')

    relation_types = ['competitive', 'complementary']
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
        diff_tag = int(node_dict[u]['tag'] != node_dict[v]['tag'])
        cnt1 += diff_tag
        rel = relation_types[diff_tag]
        f1.write(u + '\t' + '_at_'.join([rel, t]) + '\t' + v + '\n')
        f1.write(v + '\t' + '_at_'.join([rel, t]) + '\t' + u + '\n')
    for uv in valid_pair_dict['time0']:
        u, v, t = uv.split('#')
        diff_tag = int(node_dict[u]['tag'] != node_dict[v]['tag'])
        cnt2 += diff_tag
        rel = relation_types[diff_tag]
        f2.write(u + '\t' + '_at_'.join([rel, t]) + '\t' + v + '\n')
        f2.write(v + '\t' + '_at_'.join([rel, t]) + '\t' + u + '\n')
    for uv in test_pair_dict['time0']:
        u, v, t = uv.split('#')
        diff_tag = int(node_dict[u]['tag'] != node_dict[v]['tag'])
        cnt3 += diff_tag
        rel = relation_types[diff_tag]
        f3.write(u + '\t' + '_at_'.join([rel, t]) + '\t' + v + '\n')
        f3.write(v + '\t' + '_at_'.join([rel, t]) + '\t' + u + '\n')
    f1.close()
    f2.close()
    f3.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, default='./data/raw/dataset_TSMC2014_TKY.csv')
    parser.add_argument('--output_path', type=str, default='./data/tokyo/')
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    data = pd.read_csv(args.input_data)
    visit_dict = load_user_visit(data)
    visit_dict_time = update_user_visit_with_time(visit_dict)
    node_dict, relation, all_pair_list_d = build_relationship(visit_dict_time)
    node_dict, train, valid, test = build_dataset(node_dict, relation, all_pair_list_d)

    save_relationship_dataset(args.output_path, node_dict, train, valid, test)