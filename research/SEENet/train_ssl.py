import argparse
import numpy as np
import paddle
import random
from dataset import LocationRelDataset
from model import SEENet
import utils

def setup_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    data = LocationRelDataset(name=args.dataset, grid_len=args.grid_len, raw_dir=args.data_dir)
    num_nodes = data.num_nodes
    train_data = data.train
    coords = data.coords
    relation_dict = data.relation_dict
    rel_list, time_list = data.rel_list, data.time_list
    num_rels = data.num_rels
    grid_dict = data.grid

    boundaries = utils.adaptive_bin_distance(args.bin_num, coords, train_data)

    log = str(args)+'\n'
    # create model
    model = SEENet(num_nodes, args.n_hidden, num_rels,
                   boundaries=boundaries,
                   num_neighbor=args.n_neighbor,
                   time_list=time_list,
                   dropout=args.dropout,
                   w_local=args.w_local,
                   w_global=args.w_global)

    evo_data, evo_labels = utils.build_dynamic_labels(train_data, relation_dict, rel_list, time_list, num_nodes)
    hop2_dict = utils.build_hop2_dict(train_data, relation_dict, rel_list, time_list, args.dataset, path_len=2)
    test_graph = utils.generate_sampled_hetero_graphs_and_labels(train_data, num_nodes, relation_dict, time_list, hop2_dict, coords=coords, test=True)
    node_id = paddle.arange(0, num_nodes).reshape([-1, 1])

    g = test_graph
    # optimizer
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=args.lr, weight_decay=args.regularization)
    model_save_file = 'output/saved_model_%s' % (args.run_name)

    # training loop
    print("start training...")
    epoch = 0
    min_loss = 1e9
    stop_epoch = 0
    while True:
        model.train()
        epoch += 1
        grid_sizes, pos_samples, neg_samples, grid_labels = utils.generate_batch_grids(grid_dict, args.global_batch_size, args.global_neg_ratio, sampling_hop=2)
        grid_sizes = paddle.to_tensor(grid_sizes).cast('int64')
        pos_samples, neg_samples = paddle.to_tensor(pos_samples).cast('int64'), paddle.to_tensor(neg_samples).cast('int64')
        grid_labels = paddle.to_tensor(grid_labels)
        
        embed = model(g, node_id)
        loss = model.get_loss(g, embed, evo_data, evo_labels, grid_sizes, pos_samples, neg_samples, grid_labels)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        print("Epoch {:04d} | Loss {:.4f}".format(epoch, train_loss))
        log += "Epoch {:04d} | Loss {:.4f} \n".format(epoch, train_loss)
        optimizer.clear_grad()

        if train_loss < min_loss:
            min_loss = train_loss
            stop_epoch = 0
            paddle.save({'model': model.state_dict(), 'epoch': epoch}, model_save_file)
        else:
            stop_epoch += 1
        if stop_epoch == 100:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trial')
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-hidden", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-neighbor", type=int, default=5)
    parser.add_argument("--n-epochs", type=int, default=2000)
    parser.add_argument("-d", "--dataset", type=str, default='tokyo')
    parser.add_argument("--pretrain-path", type=str, default='')
    parser.add_argument("--w-local", type=float, default=1.0)
    parser.add_argument("--w-global", type=float, default=1.0)
    parser.add_argument("--grid-len", type=int, default=300)
    parser.add_argument("--global-neg-ratio", type=int, default=3)
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--negative-sample", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--bin_num", type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    setup_seed(args.seed)
    print(args)

    if int(args.gpu) == -1:
        paddle.set_device('cpu')
    else:
        paddle.set_device('gpu:%s' % args.gpu)

    run_name = f'ssl'
    run_name = args.name + "_" + run_name
    args.run_name = f'{args.dataset}_{run_name}'
    main(args)

