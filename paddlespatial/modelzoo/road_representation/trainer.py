import os
import time
from logging import getLogger
import paddle
import numpy as np
from util import ensure_dir, masked_rmse_paddle, masked_mae_paddle, masked_mape_paddle


class BaseRoadRepTrainer:
    def __init__(self, config, model, data_feature):
        self.config = config
        self.data_feature = data_feature
        self.device = self.config.get('device')
        self.model = model.to(self.device)

        self.cache_dir = './cache/model_cache'
        ensure_dir(self.cache_dir)
        self.evaluate_cache_dir = './cache/evaluate_cache'
        ensure_dir(self.evaluate_cache_dir)

        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        self._logger.info(self.model)
        for name, param in self.model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' + str(param.place))
        total_num = sum([param.size for param in self.model.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))

        self.epochs = self.config.get('max_epoch', 2)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.weight_decay = self.config.get('weight_decay', 0.01)
        self.use_early_stop = self.config.get('use_early_stop', False)
        self.patience = self.config.get('patience', 50)
        self.log_every = self.config.get('log_every', 1)
        self.saved = self.config.get('saved_model', True)

        self.output_dim = self.config.get('output_dim', 1)
        self.optimizer = paddle.optimizer.Adam(parameters=self.model.parameters(), learning_rate=self.learning_rate,
                                               weight_decay=self.weight_decay)

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        ensure_dir(self.cache_dir)
        self._logger.info("Saved model at " + cache_name)
        paddle.save((self.model.state_dict(), self.optimizer.state_dict()), cache_name)

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        self._logger.info("Loaded model at " + cache_name)
        model_state, optimizer_state = paddle.load(cache_name)
        self.model.set_state_dict(model_state)

    def save_model_with_epoch(self, epoch):
        """
        保存某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        ensure_dir(self.cache_dir)
        config = dict()
        config['model_state_dict'] = self.model.state_dict()
        config['optimizer_state_dict'] = self.optimizer.state_dict()
        config['epoch'] = epoch
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        paddle.save(config, model_path)
        self._logger.info("Saved model at {}".format(epoch))
        return model_path

    def load_model_with_epoch(self, epoch):
        """
        加载某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % epoch
        checkpoint = paddle.load(model_path)
        self.model.set_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch))

    def evaluate(self, test_dataloader):
        """
        use model to test data
        """
        pass

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config
        """
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []

        for epoch_idx in range(0, self.epochs):
            start_time = time.time()
            train_loss = self._train_epoch(train_dataloader)
            t1 = time.time()
            train_time.append(t1 - start_time)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = self._valid_epoch(eval_dataloader)
            end_time = time.time()
            eval_time.append(end_time - t2)

            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.get_lr()
                message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'.\
                    format(epoch_idx, self.epochs, np.mean(train_loss), val_loss, log_lr, (end_time - start_time))
                self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break
        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        self.load_model_with_epoch(best_epoch)
        save_list = os.listdir(self.cache_dir)
        for save_file in save_list:
            if '.tar' in save_file:
                os.remove(self.cache_dir + '/' + save_file)
        return min_val_loss

    def _train_epoch(self, train_dataloader):
        """
        完成模型一个轮次的训练

        Returns:
            float: 训练集的损失值
        """
        pass

    def _valid_epoch(self, eval_dataloader):
        """
        完成模型一个轮次的评估

        Args:
            eval_dataloader: 评估数据

        Returns:
            float: 验证集的损失值
        """
        pass


class TransductiveTrainer(BaseRoadRepTrainer):
    def __init__(self, config, model, data_feature):
        super(TransductiveTrainer, self).__init__(config, model, data_feature)

    def evaluate(self, test_dataloader):
        """
        use model to test data
        """
        node_features = paddle.to_tensor(test_dataloader['node_features'], place=self.device)
        node_labels = node_features.clone()
        test_mask = test_dataloader['mask']

        self._logger.info('Start evaluating ...')
        with paddle.no_grad():
            self.model.eval()
            output = self.model({'node_features': node_features})
            output = self._scaler.inverse_transform(output)
            node_labels = self._scaler.inverse_transform(node_labels)
            rmse = masked_rmse_paddle(output[test_mask], node_labels[test_mask])
            mae = masked_mae_paddle(output[test_mask], node_labels[test_mask])
            mape = masked_mape_paddle(output[test_mask], node_labels[test_mask])
            self._logger.info('MAE={}, RMSE={}'.format(mae.item(), rmse.item()))
            return mae.item(), mape.item(), rmse.item()

    def _train_epoch(self, train_dataloader):
        """
        完成模型一个轮次的训练

        Returns:
            float: 训练集的损失值
        """
        node_features = paddle.to_tensor(train_dataloader['node_features'], place=self.device)
        node_labels = node_features.clone()
        train_mask = train_dataloader['mask']

        self.model.train()
        loss_func = self.model.calculate_loss
        loss = loss_func({'node_features': node_features, 'node_labels': node_labels, 'mask': train_mask})
        loss.backward()
        self.optimizer.step()
        self.optimizer.clear_grad()
        return loss.item()

    def _valid_epoch(self, eval_dataloader):
        """
        完成模型一个轮次的评估

        Args:
            eval_dataloader: 评估数据

        Returns:
            float: 验证集的损失值
        """
        node_features = paddle.to_tensor(eval_dataloader['node_features'], place=self.device)
        node_labels = node_features.clone()
        valid_mask = eval_dataloader['mask']

        with paddle.no_grad():
            self.model.eval()
            loss_func = self.model.calculate_loss
            loss = loss_func({'node_features': node_features, 'node_labels': node_labels, 'mask': valid_mask})
            return loss.item()


class LINETrainer(BaseRoadRepTrainer):
    def __init__(self, config, model, data_feature):
        super(LINETrainer, self).__init__(config, model, data_feature)

    def evaluate(self, test_dataloader):
        """
        use model to test data
        """
        self._logger.info('Start evaluating ...')
        mean_loss = self._valid_epoch(test_dataloader)
        self._logger.info('mean loss={}'.format(mean_loss))
        return mean_loss

    def _train_epoch(self, train_dataloader):
        """
        完成模型一个轮次的训练

        Args:
            train_dataloader: 训练数据

        Returns:
            list: 每个batch的损失的数组
        """
        self.model.train()
        loss_func = self.model.calculate_loss
        losses = []
        for batch in train_dataloader:
            I, J, Neg = batch
            I = paddle.to_tensor(I, place=self.device)
            J = paddle.to_tensor(J, place=self.device)
            Neg = paddle.to_tensor(Neg, place=self.device)
            batch = {'I': I, 'J': J, 'Neg': Neg}
            loss = loss_func(batch)
            self._logger.debug(loss.item())
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()
        return losses

    def _valid_epoch(self, eval_dataloader):
        """
        完成模型一个轮次的评估

        Args:
            eval_dataloader: 评估数据

        Returns:
            float: 评估数据的平均损失值
        """
        with paddle.no_grad():
            self.model.eval()
            loss_func = self.model.calculate_loss
            losses = []
            for batch in eval_dataloader:
                I, J, Neg = batch
                I = paddle.to_tensor(I, place=self.device)
                J = paddle.to_tensor(J, place=self.device)
                Neg = paddle.to_tensor(Neg, place=self.device)
                batch = {'I': I, 'J': J, 'Neg': Neg}
                loss = loss_func(batch)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            return mean_loss


class GensimTrainer(BaseRoadRepTrainer):
    def __init__(self, config, model, data_feature):
        self.config = config
        self.data_feature = data_feature
        self.model = model

        self.cache_dir = './cache/model_cache'
        ensure_dir(self.cache_dir)
        self.evaluate_cache_dir = './cache/evaluate_cache'
        ensure_dir(self.evaluate_cache_dir)

    def train(self, train_dataloader, eval_dataloader):
        """
        use data to train model with config
        """
        self.model.run()

    def load_model(self, cache_name):
        pass

    def save_model(self, cache_name):
        pass
