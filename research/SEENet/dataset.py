import os
import paddle
import numpy as np


class LocationRelDataset(object):
    """Location relationship dataset
    name: str
        The name of the dataset (e.g., beijing).
    raw_dir : str
        Raw file directory to load the input data directory.
    grid_len: str
        Gridding size for global spatial infomax
    verbose: bool
      Whether to print out progress information. Default: True.
    """
    def __init__(self, name, raw_dir, grid_len, verbose=True):
        self.name = name
        self.grid_len = grid_len
        self.verbose = verbose
        self.raw_path = os.path.join(raw_dir, self.name)
        self.process()

    def process(self):
        """
        Load and process the location relationship
        """
        root_path = self.raw_path
        entity_path = os.path.join(root_path, 'entities.dict')
        relation_path = os.path.join(root_path, 'relations.dict')
        grid_path = os.path.join(root_path, 'grids_%d.dict' % self.grid_len)
        coord_path = os.path.join(root_path, 'coords.dict')
        train_path = os.path.join(root_path, 'train.txt')
        valid_path = os.path.join(root_path, 'valid.txt')
        test_path = os.path.join(root_path, 'test.txt')
        entity_dict = _read_dictionary(entity_path)
        relation_dict, rel_list, time_list = _read_relations(relation_path)
        coords = _read_coord(coord_path, entity_dict)
        grid = _read_grids_as_dict(grid_path, entity_dict)
        train = np.asarray(_read_triplets_as_list(train_path, entity_dict, relation_dict))
        valid = np.asarray(_read_triplets_as_list(valid_path, entity_dict, relation_dict))
        test = np.asarray(_read_triplets_as_list(test_path, entity_dict, relation_dict))
        num_nodes = len(entity_dict)
        num_rels = len(relation_dict)
        if self.verbose:
            print("# entities: {}".format(num_nodes))
            print("# relations: {}".format(num_rels))
            print("# training edges: {}".format(train.shape[0]))
            print("# validation edges: {}".format(valid.shape[0]))
            print("# testing edges: {}".format(test.shape[0]))

        self._relation_dict = relation_dict
        self._rel_list = rel_list
        self._time_list = time_list
        self._train = train
        self._valid = valid
        self._test = test
        self._grid = grid

        self._coords = coords
        self._num_nodes = num_nodes
        self._num_rels = num_rels
        self._entity_dict = entity_dict


    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return self._num_nodes

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def relation_dict(self):
        return self._relation_dict

    @property
    def rel_list(self):
        return self._rel_list
    
    @property
    def time_list(self):
        return self._time_list

    @property
    def coords(self):
        return self._coords

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def train(self):
        return self._train

    @property
    def valid(self):
        return self._valid

    @property
    def test(self):
        return self._test
    
    @property
    def grid(self):
        return self._grid

def _read_grids_as_dict(filename, entity_dic):
    dd = {}
    with open(filename) as f:
        for line in f.readlines():
            cont = line.strip().split('\t')
            key = tuple(map(int, cont[0].split(',')))
            dd[key] = [entity_dic[bid] for bid in cont[1:]]
    return dd

def _read_coord(filename, entity_dict):
    coords = [[] for _ in range(len(entity_dict))]
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            coords[entity_dict[line[0]]] = [float(line[1]), float(line[2])]
    return paddle.to_tensor(coords)

def _read_dictionary(filename):
    d = {}
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d

def _read_relations(filename):
    d, rs, ts = {}, set(), []
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
            r, t = line[1].split('_at_')
            rs.add(r)
            if t not in ts:
                ts.append(t)
    return d, list(rs), ts

def _read_triplets(filename):
    with open(filename) as f:
        for line in f.readlines():
            processed_line = line.strip().split('\t')
            yield processed_line

def _read_triplets_as_list(filename, entity_dict, relation_dict):
    l = []
    for triplet in _read_triplets(filename):
        s = entity_dict[triplet[0]]
        r = relation_dict[triplet[1]]
        o = entity_dict[triplet[2]]
        l.append([s, r, o])
    return l