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
    "epoch": 30,
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
trained 30 epoches, best result saved
epoch:24 steps:2290 best_valid_acc:0.9431524547803618
	Loss: 0.0052(test)	|	Acc: 95.1%(test)
```


### Result


### Conclusion
