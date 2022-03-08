# Perdict_novel
use RNN model predict novels.
The default novel is a chinese novel named '遮天'.

### Environment
```bash
pip install -r requirements.txt
```

### Save Model
```bash
python --save_model <save_path>
```

### Load Model
```bash
python --load_model <model_path>
```

### 待做 TODO
1. 在read_novel中，使用更精细的方式对文本进行预处理
2. read_novel中，打开文件的编码方式为`gbk`，是否需要对英文文本进行处理`utf-8`
3. 添加网络下载小说
4. 处理`DataParallel`的问题(训练速度慢，比单GPU慢)，使用`Distributed Data parallel`训练更大的模型
5. jieba分词结果对模型的影响研究(可能需要把词汇表修改一下)