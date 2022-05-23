import paddle
import utils


class State():
    def __init__(self, poi, trg, time_limit, trsf_matrix, device, pretraining_gen):
        super().__init__()
        node_size, batch_size = poi.shape
        self.trsf_matrix = trsf_matrix
        # self.time_remain = torch.tensor(time_limit, dtype=torch.float, device=device)
        self.time_remain = paddle.to_tensor(time_limit, dtype='float32')

        self.interval = 15
        if pretraining_gen:
            trg_len = [len(t) for t in trg]
            self.max_len = max(trg_len)
        else:
            self.max_len = 10

        # self.trg_len = torch.tensor([len(seq) for seq in trg]).to(device)

        # self.ori_poi = poi.t().cpu()  # b, L
        self.ori_poi = poi.t()
        # self.prev_a = torch.zeros(batch_size, device=device)  # b
        self.prev_a = paddle.zeros([batch_size], dtype='int64')
        # self.real_prev_a = self.ori_poi.gather(1, self.prev_a[:, None].long()).squeeze(-1)  # b
        # self.real_prev_a = self.ori_poi.gather(self.prev_a[:, None], axis=1).squeeze(-1)
        self.real_prev_a = utils.gather(self.ori_poi, 1, self.prev_a[:, None]).squeeze(-1)

        self.current_trsf = self.trsf_matrix[self.real_prev_a]
        # self.instance_current_trsf = self.current_trsf.gather(1, self.ori_poi)
        # self.instance_current_trsf = self.current_trsf.gather(self.ori_poi, axis=1)
        self.instance_current_trsf = utils.gather(self.current_trsf, 1, self.ori_poi)
        # self.instance_current_trsf[:, 0] = 9999999  # 没用

        # self.visited = torch.cat([torch.ones(batch_size, 1, device=device),
        #                          torch.zeros(batch_size, node_size - 1, device=device)], dim=-1)  # b, L
        self.visited = paddle.concat([paddle.ones((batch_size, 1)),
                                      paddle.zeros((batch_size, node_size - 1))], axis=-1)
        self.step = 1

        self.device = device

    def get_current_node(self):
        return self.prev_a

    def all_finished(self):
        return ((self.prev_a == 0).all() and (self.step > 1)) or (self.step >= self.max_len)
        # return (self.step == self.trg_len).all().item()

    def update_state(self, selected):
        # selected = selected.to(torch.long)
        selected = selected.cast('int64')
        # real_selected = self.ori_poi.gather(1, selected).squeeze(-1)  # (b, )
        # real_selected = self.ori_poi.gather(selected, axis=1).squeeze(-1)  # (b, )
        real_selected = utils.gather(self.ori_poi, 1, selected).squeeze(-1)  # (b, )
        time_cost = self.trsf_matrix[self.real_prev_a, real_selected]
        time_cost[selected.squeeze(-1) == 0] = 0
        self.time_remain -= time_cost
        self.step += 1
        # self.visited[:, selected.squeeze(-1)] = 1  # ??
        # self.visited.scatter_(1, selected, 1)
        # self.visited.scatter_(selected, axis=1, update=1)
        self.visited[paddle.arange(selected.shape[0]), selected.squeeze()] = 1
        self.prev_a = selected.squeeze(-1)
        # self.real_prev_a = self.ori_poi.gather(1, self.prev_a[:, None]).squeeze(-1)
        self.real_prev_a = utils.gather(self.ori_poi, 1, self.prev_a[:, None]).squeeze(-1)

        self.current_trsf = self.trsf_matrix[self.real_prev_a]
        # self.instance_current_trsf = self.current_trsf.gather(1, self.ori_poi)
        # self.instance_current_trsf = self.current_trsf.gather(self.ori_poi, axis=1)
        self.instance_current_trsf = utils.gather(self.current_trsf, 1, self.ori_poi)

        # self.instance_current_trsf[:, 0] = 9999999  # 不会主动回到起点
        # selected = selected.to(torch.long)
        # # unfinished = unfinished.to(torch.long)
        # real_prev_a = self.ori_poi.gather(1, self.prev_a[unfinished][:]).squeeze(-1)
        # real_selected = self.ori_poi.gather(1, selected).squeeze(-1)
        # time_cost = self.trsf_matrix[real_prev_a, real_selected]
        # # sub_trsf_matrix = self.trsf_matrix[selected]
        # # instance_trsf_matrix = sub_trsf_matrix.gather(1, self.ori_poi)
        # self.time_remain[unfinished] = self.time_remain[unfinished] - time_cost
        # self.step[unfinished] = self.step[unfinished] + 1
        # self.visited[unfinished, selected] = 1
        # self.prev_a[unfinished] = selected.squeeze(-1)

        # # jugde whether finished
        # sub_trsf_matrix = self.trsf_matrix[self.prev_a[unfinished]]
        # instance_trsf_matrix = sub_trsf_matrix.gather(1, self.ori_poi[unfinished])
        # instance_trsf_matrix[self.visited.bool()] = 9999999
        # self.unfinished[unfinished][self.time_remain[unfinished] <= instance_trsf_matrix.min(dim=1)] = 0
        # # self.unfinished[self.step == self.trg_len] = 0
        # self.prev_a[unfinished] = selected.squeeze(-1)
        # return instance_trsf_matrix

    def get_finished(self):
        # return ((self.prev_a == 0) & (self.step > 1)) | (self.step >= self.max_len)
        return ((self.prev_a == 0) & (paddle.to_tensor(self.step > 1))) | (paddle.to_tensor(self.step >= self.max_len))
        # return self.time_remain >= self.instance_current_trsf.min(dim=1)[0]  # test
        # return torch.nonzero(self.unfinished, as_tuple=False).view(-1)

    def get_time_remian(self):
        return self.time_remain
        # return self.time_remain.to(self.device)

    def get_discret_time_remain(self):
        discret_time_remain = self.time_remain / self.interval
        # discret_time_remain = discret_time_remain.cast('int64')
        return discret_time_remain

    def get_mask(self):
        # current_trsf = self.trsf_matrix[self.real_prev_a]
        # instance_current_trsf = current_trsf.gather(1, self.ori_poi)
        # instance_current_trsf[: 0] = 0
        mask = self.instance_current_trsf >= self.time_remain[:, None]
        # mask = mask | self.visited.bool()
        mask = mask | self.visited.cast('bool')
        mask[:, 0] = True
        if self.get_finished().any():
            mask[self.get_finished(), 1:] = True
            mask[self.get_finished(), 0] = False
        mask[mask.all(axis=1), 0] = False
        return mask
        # return mask.to(self.device)

        # sub_trsf_matrix = self.trsf_matrix[self.prev_a[unfinished]]
        # instance_trsf_matrix = sub_trsf_matrix.gather(1, self.ori_poi[unfinished])
        # self.visited[unfinished][instance_trsf_matrix <= self.time_remain] = 1
        # return self.visited
