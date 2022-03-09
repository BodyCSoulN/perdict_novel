import argparse
from gc import callbacks
import json

import torch
from torch import nn

from model import RNNModel
from train import train_novel
from train import predict_novel
from data_preprocess import load_data_novel


# def main(args):
    
#     batch_size, time_steps, max_tokens = args.batch_size, args.num_steps, args.max_token
#     token, language = args.token, args.language

#     train_iter, vocab = load_data_novel(batch_size, time_steps, token, language, max_tokens)

#     vocab_size, num_hiddens, num_layers = len(vocab), args.num_hiddens, args.num_layers
#     lr, num_epochs = args.lr, args.num_epochs
#     device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
#     net = args.net
#     if net == 'GRU':
#         rnn_layer = nn.GRU(vocab_size, num_hiddens, num_layers)
#     elif net == 'LSTM':
#         rnn_layer = nn.LSTM(vocab_size, num_hiddens, num_layers)
#     else:
#         print('unrecognized net, use GRU as default')
#         rnn_layer = nn.GRU(vocab_size, num_hiddens, num_layers)
    
#     model = RNNModel(rnn_layer, vocab_size)

#     # model = nn.DataParallel(model, device_ids=[0, 1]).to(device)
#     model = model.to(device)
#     load_model = args.load_model
#     if load_model:
#         try:
#             model_state_dict = torch.load(load_model)
#             model.load_state_dict(model_state_dict)
#         except Exception as e:
#             print('load model error', e)
#             return
#     else:
#         train_novel(model, train_iter, vocab, lr, num_epochs, device)

#     predict = lambda prefix: predict_novel(prefix, 1000, model, vocab, device)
#     predict('叶凡')

#     save_model = args.save_model
#     if save_model:
#         try:
#             torch.save(model.state_dict(), save_model)
#         except Exception as e:
#             print('save model error', e)
#             print('the state dict of model is\n', model.state_dict())

def to_train(args):
    batch_size, time_steps, max_tokens = args.batch_size, args.num_steps, args.max_token
    token, language = args.token, args.language

    train_iter, vocab = load_data_novel(batch_size, time_steps, token, language, max_tokens)

    vocab_size, num_hiddens, num_layers = len(vocab), args.num_hiddens, args.num_layers
    lr, num_epochs = args.lr, args.num_epochs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    load_model_path = args.load_model_path
    if load_model_path:
        try:
            model_state_dict = torch.load(load_model_path)
            model.load_state_dict(model_state_dict)
        except Exception as e:
            print('load model error', e)
            return
    train_novel(model, train_iter, vocab, lr, num_epochs, device)
    with open('model/train_args' + '.json', 'w', encoding='utf-8') as f:
        json.dump(args.__dict__, f, ensure_ascii=False)
    save_model_path = args.save_model_path
    if save_model_path:
        try:
            torch.save(model.state_dict(), save_model_path)
        except Exception as e:
            print('save model error', e)
            print('the state dict of model is\n', model.state_dict())

def to_predict(args):
    prefix = args.prefix
    num_preds = args.num_preds
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open('model/train_args.json', 'r', encoding='utf-8') as f:
        train_args = json.load(f)
    
    _, vocab = load_data_novel(train_args['batch_size'], train_args['time_steps'], train_args['token'], train_args['language'], train_args['max_tokens'])
    if train_args['net'] == 'GRU':
        rnn_layer = nn.GRU(train_args['vocab_size'], train_args['num_hiddens'], train_args['num_layers'])
    elif train_args['net'] == 'LSTM':
        rnn_layer = nn.LSTM(train_args['vocab_size'], train_args['num_hiddens'], train_args['num_layers'])
    else:
        print('unrecognized net, use GRU as default')
        rnn_layer = nn.GRU(train_args['vocab_size'], train_args['num_hiddens'], train_args['num_layers'])
    model = RNNModel(rnn_layer, len(vocab))
    load_model_path = args.load_model_path
    if load_model_path:
        try:
            model_state_dict = torch.load(load_model_path)
            model.load_state_dict(model_state_dict)
        except Exception as e:
            print('load model error', e)
            return
    predict_novel(prefix, num_preds, model, vocab, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()

    train_parser = subparser.add_parser('train', help='training the model')
    train_parser.add_argument('--language', type=str, default='chinese')
    train_parser.add_argument('--token', type=str, default='char')
    train_parser.add_argument('--net', type=str, default='GRU', help='which rnn to use GRU/LSTM')
    train_parser.add_argument('--batch_size', type=int, default=256)
    train_parser.add_argument('--num_steps', type=int, default=35)
    train_parser.add_argument('--max_token', type=int, default=1000000, help='how many tokens for training')    
    train_parser.add_argument('--num_hiddens', type=int, default=256)    
    train_parser.add_argument('--num_layers', type=int, default=1)    
    train_parser.add_argument('--num_epochs', type=int, default=500)
    train_parser.add_argument('--lr', type=float, default=1e-3) 
    train_parser.add_argument('--load_model_path', type=str, default=None)
    train_parser.add_argument('--save_model_path', type=str, default=None)
    train_parser.set_defaults(action='train')

    predict_parser = subparser.add_parser('predict', help='predict the prefix')
    predict_parser.add_argument('--load_model_path', type=str, default=None, required=True)
    predict_parser.add_argument('--prefix', type=str, default=None, required=True)
    predict_parser.add_argument('--num_preds', type=int, default=100)
    predict_parser.set_defaults(action='predict')

    args = parser.parse_args()

    if args.action == 'train':
        to_train(args)
    elif args.action == 'predict':
        to_predict(args)
    # main(args)