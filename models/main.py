import torch
import data_loader
import transformer_model
import lstm_model
import argparse
import torch.optim as optim
import time
from data_utils import batchify
import sys
import os
from torch.optim.adam import Adam
import json

parser = argparse.ArgumentParser(description = 'parameters for DiscreteSeq project')
parser.add_argument('--dataset_path', type=str, default = '../../dataset/',
                    help = 'the location of the dataset')

parser.add_argument('--model', type=str, default = 'transformer',
                    help = 'the type of model')
parser.add_argument('--q_type', type=str, default = 'Task_1',
                    help = 'the task type')
parser.add_argument('--batch_size', type=int, default = 10,
                    help = 'the size of the batches')
parser.add_argument('--nhid', type = int, default = 100,
                    help = 'the hidden state size')
parser.add_argument('--self_attn', type=int, default = 1,
                    help = 'if we apply self attention in Transformer')
parser.add_argument('--nheads', type=int, default = 1,
                    help = 'the number of attention heads')
parser.add_argument('--nlayers', type=int, default = 1,
                    help = 'number of transformer layers')
parser.add_argument('--optimizer', type = str, default = 'Adam',
                    help = 'the optimizer used for the model')
parser.add_argument('--lr', type = float, default = 0.0001,
                    help = 'the learning rate used for backprop')
parser.add_argument('--nb_epochs', type = int, default = 10,
                    help = 'the number of training epochs')
parser.add_argument('--clip', type = float, default = 0.5,
                    help = 'clipping the gradient')
parser.add_argument('--dropout', type = float, default = 0.2,
                    help = 'dropout applied to rnn')
parser.add_argument('--cuda', action = 'store_true',
                    help = 'use cuda')
parser.add_argument('--seq_size', type = int, default = 5,
                    help = 'length of sequence')
parser.add_argument('--nonlin', type = str, default = 'relu',
                    help = 'nonlinearity applied to concatenation')
parser.add_argument('--query_nonlin', type = str, default = 'no',
                    help = 'query embedding nonlinearity')
parser.add_argument('--query_attn', type = int, default = 1,
                    help = 'if we use query attention')
parser.add_argument('--pos_encoding', type = int, default = 1,
                    help = 'if transformer contains pos encoding')
parser.add_argument('--feature_size', type = int, default = 36,
                    help = 'the number of features for the input')
parser.add_argument('--log_file', type = str, default = 'test.txt',
                    help = 'where to write the logs')
parser.add_argument('--check_interval', type = int, default = 1,
                    help = 'number of epochs to run validation')
parser.add_argument('--save', type=str, default = 'model.pt',
                    help = 'the file to save the best model')
parser.add_argument('--seed', type=int, default = 1111,
                    help = 'random seed')
parser.add_argument('--slurm', action = 'store_true',
                    help = 'if slurm is used to run the script')
parser.add_argument('--config_file', type=str, default='config.ini',
                    help = 'the path for the argparse arguments')
parser.add_argument('--attn_file', type = str, default = 'attn_weights.tsv',
                    help = 'where to write the logs')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('Warning: You have a CUDA device, you should use it')

device = torch.device("cuda" if args.cuda else "cpu")

args.dataset_path += args.q_type + '/' + str(args.seq_size) + 'len/'

if args.slurm:
    slurm_id = os.environ.get('SLURM_JOB_ID')
    if not os.path.isdir('logs/' + str(slurm_id)):
        os.makedirs('logs/' + str(slurm_id))
    args.save = 'logs/' + str(slurm_id) + '/' + args.save
    args.log_file = 'logs/' + str(slurm_id) + '/' + args.log_file
    args.config_file = 'logs/' + str(slurm_id) + '/' + args.config_file
    args.attn_file = 'logs/' + str(slurm_id) + '/' + args.attn_file

with open(args.config_file, 'w') as f:
    json.dump(args.__dict__, f, indent=2)


def write_logs(out, target, out_file):
    for i in range(len(out)):
        out_file.write('{:1.4f}\t{:1.4f}\n'.format(out[i].item(), target[i].item()))

def write_attn(attn_w, out_file):
    weights = attn_w.cpu().detach().squeeze(2).numpy()
    for i in range(len(weights)):
        out_file.write('\t'.join(str(x) for x in weights[i]))
        out_file.write('\n')

def get_accuracy(out, target):
    correct = 0.0
    total = 0.0
    for i in range(len(out)):
        total += 1
        if ((out[i].item() >= 0.5 and target[i].item() == 1) or
            (out[i].item() < 0.5 and target[i].item() == 0)):
                correct += 1
    return correct, total


if __name__ == '__main__':
    dataset = data_loader.SeqData(args.dataset_path, args.feature_size, args.q_type)
    if args.model == 'transformer':
        model = transformer_model.Transformer(dataset.inp_emb_size, args.nheads,
                args.nhid, args.nlayers, args.nonlin, args.dropout, args.self_attn, args.query_attn, args.pos_encoding).to(device)
    elif args.model == 'lstm':
        model = lstm_model.Lstm(dataset.inp_emb_size, dataset.inp_emb_size,
                args.nhid, 'LSTM', args.nlayers, args.nonlin, args.query_attn, args.dropout).to(device)
    params = filter(lambda p: p.requires_grad, model.parameters())
    opt = getattr(optim, args.optimizer)(params, lr = args.lr)
    b_size = args.batch_size
    model.train()
    best_acc = 0.0
    for epoch in range(0, args.nb_epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        out_file = open(args.log_file, 'w')
        total = 0
        correct = 0
        for i in range(0, len(dataset.training), b_size):
            inputs, q, target = batchify(dataset.training[i:i + b_size], b_size, device)
            opt.zero_grad()
            if args.model == 'transformer':
                inputs = inputs.permute(1,0,2)
                out, _ = model(inputs, q)
            elif args.model == 'lstm':
                hidden = model.init_hidden(b_size)
                inputs = inputs.permute(1,0,2)
                out, hidden, _ = model(inputs, hidden, q)
            loss = torch.nn.BCELoss()(out, target.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total_loss += loss.data.item()
            local_correct, local_total = get_accuracy(out, target)
            correct += local_correct
            total += local_total
        elapsed = time.time() - start_time
        nr_batch = len(dataset.training) / b_size
        if epoch % args.check_interval == 0:
            model.eval()
            val_total, val_correct = 0, 0
            val_total_loss = 0.0
            for i in range(0, len(dataset.valid), b_size):
                inputs, q, target = batchify(dataset.valid[i:i+b_size], b_size, device)
                if args.model == 'transformer':
                    inputs = inputs.permute(1,0,2)
                    out, _ = model(inputs, q)
                elif args.model == 'lstm':
                    hidden = model.init_hidden(b_size)
                    inputs = inputs.permute(1,0,2)
                    out, hidden, _ = model(inputs, hidden, q)
                loss = torch.nn.BCELoss()(out, target.squeeze())
                val_total_loss += loss.data.item()
                local_correct, local_total = get_accuracy(out, target.squeeze())
                val_correct += local_correct
                val_total += local_total
            val_acc = val_correct / val_total
            if best_acc < val_acc:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_acc = val_acc
            print('| epoch {:3d} | lr {:02.6f} | ms/batch {:5.2f} | loss {:5.2f} | accuracy {:2.5f} | val acc {:2.5f}|'.format(
                epoch, args.lr, 1000 * elapsed / nr_batch, total_loss / nr_batch, correct / total, val_acc))
    with open(args.save, 'rb') as f:
        model = torch.load(f)
        model.eval()
        test_total, test_correct = 0,0
        test_total_loss = 0.0
        log_file = open(args.log_file, 'w')
        if args.query_attn:
            attn_file = open(args.attn_file, 'w')
        for i in range(0, len(dataset.test), b_size):
            inputs, q, target = batchify(dataset.test[i:i+b_size], b_size, device)
            if args.model == 'transformer':
                inputs = inputs.permute(1,0,2)
                out, attn_w = model(inputs, q)
            elif args.model == 'lstm':
                hidden = model.init_hidden(b_size)
                inputs = inputs.permute(1,0,2)
                out, hidden, attn_w = model(inputs, hidden, q)
            loss = torch.nn.BCELoss()(out, target.squeeze())
            test_total_loss += loss.data.item()
            local_correct, local_total = get_accuracy(out, target.squeeze())
            write_logs(out, target.squeeze(), log_file)
            if args.query_attn:
                write_attn(attn_w, attn_file)
            test_correct += local_correct
            test_total += local_total
        test_acc = test_correct / test_total
        test_loss = test_total_loss / (len(dataset.test) / b_size)
        print('test results: | test loss {:5.2f} | test acc {:2.5f}|'.format(
            test_loss, test_acc))
