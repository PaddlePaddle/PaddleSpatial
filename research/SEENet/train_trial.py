import argparse
import numpy as np
import paddle
import random
from dataset import LocationRelDataset
from model import SEENetPred
import utils

def setup_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    data = LocationRelDataset(name=args.dataset,grid_len=args.grid_len,raw_dir=args.data_dir)
    num_nodes = data.num_nodes
    train_data, valid_data, test_data = data.train, data.valid, data.test
    coords = data.coords
    relation_dict = data.relation_dict
    rel_list, time_list = data.rel_list, data.time_list
    num_rels = data.num_rels

    boundaries = utils.adaptive_bin_distance(args.bin_num, coords, train_data)

    log = str(args)+'\n'
    # create model
    model = SEENetPred(num_nodes, args.n_hidden, num_rels,
                       boundaries=boundaries,
                       num_neighbor=args.n_neighbor,
                       time_list=time_list,
                       dropout=args.dropout)

    if args.pretrain_path:
        checkpoint = paddle.load(args.pretrain_path)
        model_now_dict = model.state_dict()
        for k, v in checkpoint['model'].items():
            if 'discriminator' in k or 'mlp' in k:
                continue
            model_now_dict[k] = v
        model.set_state_dict(model_now_dict)
        print("Loading pre-trained model for epoch: {}".format(checkpoint['epoch']))

    # validation and testing triplets
    valid_data = paddle.to_tensor(valid_data)
    test_data = paddle.to_tensor(test_data)

    hop2_dict = utils.build_hop2_dict(train_data, relation_dict, rel_list, time_list, args.dataset, path_len=2)
    test_graph = utils.generate_sampled_hetero_graphs_and_labels(train_data, num_nodes, relation_dict, time_list, hop2_dict, coords=coords, test=True)
    node_id = paddle.arange(0, num_nodes).reshape([-1, 1])

    valid_data_list = utils.select_time_data(valid_data, relation_dict, time_list)
    test_data_list = utils.select_time_data(test_data, relation_dict, time_list)
    train_data_list = utils.select_time_data(paddle.to_tensor(train_data), relation_dict, time_list)

    # optimizer
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=args.lr, weight_decay=args.regularization)
    model_state_file = 'runs/saved_model_%s' % (args.run_name)

    # training loop
    print("start training...")
    epoch = 0
    best_mrr = 0
    best_metrics = 0
    while True:
        model.train()
        epoch += 1
        g, data, labels = utils.generate_sampled_hetero_graphs_and_labels(train_data, num_nodes, relation_dict, time_list,
                                                                         hop2_dict=hop2_dict,
                                                                         coords=coords,
                                                                         negative_rate=args.negative_sample,
                                                                         split_size=0.8)

        data, labels = paddle.to_tensor(data), paddle.to_tensor(labels).cast('int64').reshape([-1,1])
        embed = model(g, node_id)
        data_labels = paddle.concat([data, labels], axis=1)
        data_labels = utils.select_time_data(data_labels, relation_dict, time_list)
        data, labels = [data_labels[i][:, :3] for i in range(len(time_list))], [data_labels[i][:, 3].cast('float32') for i in range(len(time_list))]
        loss = model.get_loss(g, embed, data, labels)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        print("Epoch {:04d} | Loss {:.4f} | Best Valid MRR@10 {:.4f}".format(epoch, train_loss, best_mrr))
        log += "Epoch {:04d} | Loss {:.4f}  | Best Valid MRR@10 {:.4f}\n".format(epoch, train_loss, best_mrr)
        optimizer.clear_grad()

        # validation
        if epoch % args.evaluate_every == 0:
            model.eval()
            valid_metrics_list = []
            with paddle.no_grad():
                embed = model(test_graph, node_id)
                for ii, (trn_data, tst_data, val_data) in enumerate(zip(train_data_list, test_data_list, valid_data_list)):
                    valid_ranks = model.rank_score_filtered(embed[ii], val_data, trn_data, tst_data)
                    valid_metrics = utils.mrr_hit_metric(valid_ranks)
                    valid_metrics_list.append(valid_metrics)
                    log += str(valid_metrics) + '\n'
            
            valid_metrics = utils.overall_mrr_hit_metric(args.data_dir, args.dataset, valid_metrics_list, time_list)
            # save best model
            if best_mrr < valid_metrics['MRR@10']:
                best_mrr = valid_metrics['MRR@10']
                paddle.save({'model': model.state_dict(), 'epoch': epoch}, model_state_file)
            
            if epoch >= args.n_epochs:
                break

    print("training done")
    print("\nstart testing:")
    # use best model checkpoint
    checkpoint = paddle.load(model_state_file)
    model.eval()
    with paddle.no_grad():
        model.set_state_dict(checkpoint['model'])
        print("Using best epoch: {}".format(checkpoint['epoch']))
        embed = model(test_graph, node_id)
        test_metrics_list = []
        for ii, (trn_data, tst_data, val_data) in enumerate(zip(train_data_list, test_data_list, valid_data_list)):
            test_ranks = model.rank_score_filtered(embed[ii], tst_data, trn_data, val_data)
            test_metrics = utils.mrr_hit_metric(test_ranks)
            test_metrics_list.append(test_metrics)
            # print(test_metrics)
            log += str(test_metrics) + '\n'
        test_metrics = utils.overall_mrr_hit_metric(args.data_dir, args.dataset, test_metrics_list, time_list)

    print_result = ''
    for k, v in test_metrics.items():
        print_result += 'Test {:s} {:.4f} | '.format(k, v)
    print(print_result)

    log += str(print_result) + '##' + args.run_name
    f = open(f'logs/{args.run_name}.txt', 'w')
    f.write(log)
    f.close()
    f = open(f'output/{args.run_name}.txt', 'w')
    f.write(log)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trial')
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-hidden", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-neighbor", type=int, default=5)
    parser.add_argument("--n-epochs", type=int, default=3000)
    parser.add_argument("-d", "--dataset", type=str, default='tokyo')
    parser.add_argument("--pretrain-path", type=str, default='')
    parser.add_argument("--grid-len", type=int, default=300)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--negative-sample", type=int, default=5)
    parser.add_argument("--evaluate-every", type=int, default=100)
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

    run_name = 'trial'
    if args.pretrain_path and args.name == 'default':
        args.name = 'ssl'
    run_name = args.name + "_" + run_name
    args.run_name = f'{args.dataset}_{run_name}'
    main(args)