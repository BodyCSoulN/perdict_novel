import math
import torch
from torch import nn
from data_preprocess import trans_dim
from d2l import torch as d2l

def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    # 梯度裁剪，比较模型的所有层的所有参数的梯度的l2长度和theta的大小，如果大，那就拉到theta，否则不动
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for p in params:
            p.grad[:] = theta / norm * p.grad

def train_epoch(model, loss, updater, train_iter, device, use_random_iter=True):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            if isinstance(model, nn.DataParallel):
                state = model.module.begin_state(X.shape[0], device)
            else:
                state = model.begin_state(X.shape[0], device)
        else:
            # 在使用多gpu训练时 _detach出错。所以仅考虑随机采样
            state._detach()
        # 变成一个1维的向量
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        # print('state shape before to model', state.shape)
        state = trans_dim(state)
        y_hat, state = model(X, state)
        # print('state shape after from model', state.shape)
        state = trans_dim(state)
        
        # y_hat.shape = y.shape * vocab_size
        # print('y_hat.shape = ', y_hat.shape
        #      +'\ny.shape = ', y.shape)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(model, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(model, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel(), l)
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop(), l
    
        
    
def train_novel(model, train_iter, vocab, lr, num_epochs, device):
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    if isinstance(model, nn.Module):
        updater = torch.optim.Adam(model.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(model.params, lr, batch_size)
    loss = nn.CrossEntropyLoss()
    predict = lambda prefix: predict_novel(prefix, 100, model, vocab, device)
    for epoch in range(num_epochs):
        ppl, speed, l = train_epoch(model, loss, updater, train_iter, device)
        if epoch % 10 == 0:
            print(f'loss: {l}')
            animator.add(epoch + 1, [ppl])
            
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒, loss: {l}')
    predict('叶凡')
    
def predict_novel(prefix, num_preds, model, vocab, device):
    if isinstance(model, nn.DataParallel):
        state = model.module.begin_state(1, device)
    else:
        state = model.begin_state(1, device)
    outputs = [vocab[prefix[0]]]
    
    get_inputs = lambda: torch.tensor([outputs[-1]], device=device).reshape(1, 1)
    
    for y in prefix[1:]:
        _, state = model(get_inputs(), state)
        outputs.append(vocab[y])
        
    for i in range(num_preds):
        y, state = model(get_inputs(), state)
        outputs.append(y.argmax(dim=1))
    return ''.join([vocab.to_token(index) for index in outputs])