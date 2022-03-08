import argparse

import torch
from torch import nn

from model import RNNModel
from train import train_novel
from train import predict_novel
from data_preprocess import load_data_novel


def main(args):
    
    batch_size, time_steps, max_tokens = args.batch_size, args.num_steps, args.max_token
    token, language = args.token, args.language

    train_iter, vocab = load_data_novel(batch_size, time_steps, token, language, max_tokens)
    vocab_size, num_hiddens, num_layers = len(vocab), args.num_hiddens, args.num_layers
    lr, num_epochs = args.lr, args.num_epochs
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    net = args.net
    if net == 'GRU':
        rnn_layer = nn.GRU(vocab_size, num_hiddens, num_layers)
    elif net == 'LSTM':
        rnn_layer = nn.LSTM(vocab_size, num_hiddens, num_layers)
    else:
        print('unrecognized net, use GRU as default')
        rnn_layer = nn.GRU(vocab_size, num_hiddens, num_layers)
    
    model = RNNModel(rnn_layer, vocab_size)

    # model = nn.DataParallel(model, device_ids=[0, 1]).to(device)
    model = model.to(device)
    load_model = args.load_model
    if load_model:
        try:
            model_state_dict = torch.load(load_model)
            model.load_state_dict(model_state_dict)
        except Exception as e:
            print('load model error', e)
            return
    else:
        train_novel(model, train_iter, vocab, lr, num_epochs, device)

    predict = lambda prefix: predict_novel(prefix, 1000, model, vocab, device)
    predict('叶凡')

    save_model = args.save_model
    if save_model:
        try:
            torch.save(model.state_dict(), save_model)
        except Exception as e:
            print('save model error', e)
            print('the state dict of model is\n', model.state_dict())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default='chinese')
    parser.add_argument('--token', type=str, default='char')
    parser.add_argument('--net', type=str, default='GRU', help='which rnn to use GRU/LSTM')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=35)
    parser.add_argument('--max_token', type=int, default=1000000, help='how many tokens for training')    
    parser.add_argument('--num_hiddens', type=int, default=256)    
    parser.add_argument('--num_layers', type=int, default=1)    
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3) 
    parser.add_argument('--save_model', type=str, default=None)
    parser.add_argument('--load_model', type=str, default=None)
    args = parser.parse_args()
    main(args)