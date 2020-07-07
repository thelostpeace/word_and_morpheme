# Word and Morpheme

This repo means to find the relationship between word and morpheme in linguistics.

In English, words may have stem and affixes, such as `beautiful => beauty`, `grateful => grate`, `reality => real`. In Chinese, `蝴蝶 => 蝴 蝶`, `开心 => 开 心`.
This repo is an experiment on Chinese words and morphemes, ngram words won't be splitted, so `蝴蝶` has it's own embedding and it's meaning is attented on embedding of `蝴` and `蝶`.
So in this example `蝴蝶` is the query and `蝴`, `蝶` are the values.

A classification is done with TextCNN with one model splitting words in morpheme level and the other in word level.

### Model

#### TextCNN with morpheme level

**config**

```
{
    "seed": 1992,
    "tensorboard": "tensorboard",
    "tensorboard_flush_sec": "30",
    "train": "data/train.txt",
    "dev": "data/dev.txt",
    "test": "data/test.txt",
    "vocab_save": "model/vocab.txt",
    "label_save": "model/label.txt",
    "lr": 3e-05,
    "eps": 1e-08,
    "weight_decay": 1e-06,
    "batch_size": 32,
    "sequence_length": 32,
    "save_model": "model/textcnn.bin",
    "epoch": 60,
    "save_step": 10,
    "embedding_size": 300,
    "embedding_droprate": 0.5,
    "filter_count": 128,
    "kernel_size": [
        1,
        2,
        3,
        4
    ],
    "conv_droprate": 0.5,
    "vocab_size": 479,
    "labels": 3
}
```

**model**

```
------------------------------------------------------------------------
      Layer (type)         Output Shape         Param #     Tr. Param #
========================================================================
       Embedding-1        [32, 32, 300]         143,700         143,700
         Dropout-2        [32, 32, 300]               0               0
          Conv2d-3     [32, 128, 32, 1]          38,528          38,528
          Conv2d-4     [32, 128, 31, 1]          76,928          76,928
          Conv2d-5     [32, 128, 30, 1]         115,328         115,328
          Conv2d-6     [32, 128, 29, 1]         153,728         153,728
       MaxPool2d-7      [32, 128, 1, 1]               0               0
       MaxPool2d-8      [32, 128, 1, 1]               0               0
       MaxPool2d-9      [32, 128, 1, 1]               0               0
      MaxPool2d-10      [32, 128, 1, 1]               0               0
        Dropout-11         [32, 1, 512]               0               0
         Linear-12           [32, 1, 3]           1,539           1,539
========================================================================
Total params: 529,751
Trainable params: 529,751
Non-trainable params: 0
------------------------------------------------------------------------
```

**result**

```
trained 60 epoches, best result at epoch 52.

epoch:52 steps:5010 best_valid_acc:0.9689922480620154
    Loss: 0.0034(test)  |   Acc: 96.7%(test)
```

#### TextCNN with word level

**result**

```
epoch:56 steps:5430 best_valid_acc:0.9483204134366925
    Loss: 0.0040(test)  |   Acc: 95.9%(test)
```

#### TextCNN with word attented on morphemes

**config**

```
{
    "seed": 1992,
    "tensorboard": "tensorboard",
    "tensorboard_flush_sec": "30",
    "train": "data/train.txt",
    "dev": "data/dev.txt",
    "test": "data/test.txt",
    "vocab_save": "model/vocab.txt",
    "label_save": "model/label.txt",
    "lr": 3e-05,
    "eps": 1e-08,
    "weight_decay": 1e-06,
    "batch_size": 32,
    "sequence_length": 32,
    "save_model": "model/textcnn.bin",
    "epoch": 60,
    "save_step": 10,
    "embedding_size": 200,
    "embedding_droprate": 0.5,
    "filter_count": 128,
    "kernel_size": [
        1,
        2,
        3,
        4
    ],
    "conv_droprate": 0.5,
    "max_ngram": 8,
    "temperature": 1,
    "vocab_size": 851,
    "labels": 3
}
```

**model**

```
------------------------------------------------------------------------
      Layer (type)         Output Shape         Param #     Tr. Param #
========================================================================
       Embedding-1        [32, 32, 200]         170,200         170,200
       Embedding-2     [32, 32, 8, 200]         170,200         170,200
         Dropout-3        [32, 32, 400]               0               0
          Conv2d-4     [32, 128, 32, 1]          51,328          51,328
          Conv2d-5     [32, 128, 31, 1]         102,528         102,528
          Conv2d-6     [32, 128, 30, 1]         153,728         153,728
          Conv2d-7     [32, 128, 29, 1]         204,928         204,928
       MaxPool2d-8      [32, 128, 1, 1]               0               0
       MaxPool2d-9      [32, 128, 1, 1]               0               0
      MaxPool2d-10      [32, 128, 1, 1]               0               0
      MaxPool2d-11      [32, 128, 1, 1]               0               0
        Dropout-12         [32, 1, 512]               0               0
         Linear-13           [32, 1, 3]           1,539           1,539
========================================================================
Total params: 854,451
Trainable params: 854,451
Non-trainable params: 0
------------------------------------------------------------------------
```

**result**

```
epoch:45 steps:4270 best_valid_acc:0.9483204134366925
	Loss: 0.0045(test)	|	Acc: 95.1%(test)
```

**word meaning attented on morhpeme**
![](https://github.com/thelostpeace/word_and_morpheme/blob/master/img/1.png?raw=true)

对于单个字，因为在计算alignment的时候mask掉了`<PAD>`，所以永远只关注自身，`对呀`更关注于`对`，`什么`更关注于`么`

![](https://github.com/thelostpeace/word_and_morpheme/blob/master/img/2.png?raw=true)

`需要`更关注于`要`

![](https://github.com/thelostpeace/word_and_morpheme/blob/master/img/3.png?raw=true)
`可以`更关注于`可`


### Conclusion

word attended on morpheme，`temperature`对于模型效果的影响很大，取决于是否想突出关注于某个词素，这样导致模型不好调参，再就是计算alignment的时候取用的`scale`不好定，这个取值也会对模型效果有很大影响，整体来说，word attented on morpheme并没有增加更多的信息，某个词素在训练集里的bias，其实字粒度切词就已经学到了。
