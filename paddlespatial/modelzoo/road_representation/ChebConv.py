import pgl
import paddle.nn as nn
import numpy as np
from logging import getLogger
import util as utils


class ChebConv(nn.Layer):
    def __init__(self, config, data_feature):
        super().__init__()
        self.adj_mx_pgl = data_feature.get('adj_mx_pgl')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)

        self.output_dim = config.get('output_dim', 32)
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self._logger = getLogger()
        self._scaler = data_feature.get('scaler')

        self.encoder = pgl.nn.GCNConv(input_size=self.feature_dim, output_size=self.output_dim)
        self.decoder = pgl.nn.GCNConv(input_size=self.output_dim, output_size=self.feature_dim)

    def forward(self, batch):
        """
        自回归任务

        Args:
            batch: dict, need key 'node_features' contains tensor shape=(N, feature_dim)

        Returns:
            torch.tensor: N, feature_dim

        """
        inputs = batch['node_features']
        encoder_state = self.encoder(self.adj_mx_pgl, inputs)  # N, output_dim
        np.save('./cache/evaluate_cache/embedding_{}_{}_{}.npy'
                .format(self.model, self.dataset, self.output_dim),
                encoder_state.detach().cpu().numpy())
        output = self.decoder(self.adj_mx_pgl, encoder_state)  # N, feature_dim
        return output

    def calculate_loss(self, batch):
        """

        Args:
            batch: dict, need key 'node_features', 'node_labels', 'mask'

        Returns:

        """
        y_true = batch['node_labels']  # N, feature_dim
        y_predicted = self.forward(batch)  # N, feature_dim
        y_true = self._scaler.inverse_transform(y_true)
        y_predicted = self._scaler.inverse_transform(y_predicted)
        mask = batch['mask']
        return utils.masked_mse_paddle(y_predicted[mask], y_true[mask])
