# Perdict_novel
A novel prediction RNN model with torch.

The default novel is a chinese novel named '遮天'.

### Environment
```bash
pip install -r requirements.txt
```

### Usage
#### Train
```bash
python main.py train --save_model_path <save_model_path>
```
#### Predict
```bash
python main.py predict --load_model_path <load_model_path> --prefix <prefix>
```
> ATTENTION: If you are using cheinese words to predict. Please modify the code in `main.py`(i.e `predict_parser.add_argument('--prefix', type=str, default="叶凡")`),

### 待做 TODO
1. 在read_novel中，使用更精细的方式对文本进行预处理
2. read_novel中，打开文件的编码方式为`gbk`，是否需要对英文文本进行处理`utf-8`
3. 添加网络下载小说
4. 处理`DataParallel`的问题(训练速度慢，比单GPU慢)，使用`Distributed Data parallel`训练更大的模型
5. jieba分词结果对模型的影响研究(可能需要把词汇表修改一下)
6. 中文进行预测的时候，无法在命令行输入。


