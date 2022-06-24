import paddle
import utils


class State():
    def __init__(self, poi, trg, time_limit, trsf_matrix, device, pretraining_gen):
        super().__init__()
        node_size, batch_size = poi.shape
        self.trsf_matrix = trsf_matrix
        self.time_remain = paddle.to_tensor(time_limit, dtype='float32')

        self.interval = 15
        if pretraining_gen:
            trg_len = [len(t) for t in trg]
            self.max_len = max(trg_len)
        else:
            self.max_len = 10

        self.ori_poi = poi.t()
        self.prev_a = paddle.zeros([batch_size], dtype='int64')
        self.real_prev_a = utils.gather(self.ori_poi, 1, self.prev_a[:, None]).squeeze(-1)

        self.current_trsf = self.trsf_matrix[self.real_prev_a]
        self.instance_current_trsf = utils.gather(self.current_trsf, 1, self.ori_poi)

        self.visited = paddle.concat([paddle.ones((batch_size, 1)),
                                      paddle.zeros((batch_size, node_size - 1))], axis=-1)
        self.step = 1
        self.device = device

    def get_current_node(self):
        return self.prev_a

    def all_finished(self):
        return ((self.prev_a == 0).all() and (self.step > 1)) or (self.step >= self.max_len)

    def update_state(self, selected):
        selected = selected.cast('int64')
        real_selected = utils.gather(self.ori_poi, 1, selected).squeeze(-1)  # (b, )
        time_cost = self.trsf_matrix[self.real_prev_a, real_selected]
        time_cost[selected.squeeze(-1) == 0] = 0
        self.time_remain -= time_cost
        self.step += 1
        self.visited[paddle.arange(selected.shape[0]), selected.squeeze()] = 1
        self.prev_a = selected.squeeze(-1)
        self.real_prev_a = utils.gather(self.ori_poi, 1, self.prev_a[:, None]).squeeze(-1)

        self.current_trsf = self.trsf_matrix[self.real_prev_a]
        self.instance_current_trsf = utils.gather(self.current_trsf, 1, self.ori_poi)

    def get_finished(self):
        return ((self.prev_a == 0) & (paddle.to_tensor(self.step > 1))) | (paddle.to_tensor(self.step >= self.max_len))

    def get_time_remian(self):
        return self.time_remain

    def get_discret_time_remain(self):
        discret_time_remain = self.time_remain / self.interval
        return discret_time_remain

    def get_mask(self):
        mask = self.instance_current_trsf >= self.time_remain[:, None]
        mask = mask | self.visited.cast('bool')
        mask[:, 0] = True
        if self.get_finished().any():
            mask[self.get_finished(), 1:] = True
            mask[self.get_finished(), 0] = False
        mask[mask.all(axis=1), 0] = False
        return mask
