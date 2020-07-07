from model import MorphemeTextCNN, Config
#from tools.tokenizer import LATokenizer
import torch
import json
from tools.data_loader import load_vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_file = "morpheme_textcnn.json"
config = Config(json.load(open(config_file)))
label2id, id2label = load_vocab(config.label_save)
word2id, id2word = load_vocab(config.vocab_save)

extend_config = {
    "vocab_size": len(word2id),
    "labels": len(label2id)
}
config.set(extend_config)
model = MorphemeTextCNN(config)
model.to(device)
saved_model = torch.load(config.save_model)
model.load_state_dict(saved_model['state_dict'])

# padding sequence with <PAD>
def padding(data, fix_length, pad, add_first="", add_last=""):
    if add_first:
        data.insert(0, add_first)
    if add_last:
        data.append(add_last)
    pad_data = []
    data_len = len(data)
    for idx in range(fix_length):
        if idx < data_len:
            pad_data.append(data[idx])
        else:
            pad_data.append(pad)
    return pad_data

def is_reserved_token(w):
    tokens = ['<PAD>', '<EOS>', '<BOS>', '<UNK>', '<MORPHEME>']
    if w in tokens:
        return True
    return False

def generate_input(batch):
    # TextDataSet yield one line contain label and input
    batch_input, batch_morpheme, batch_attention_mask = [], [], []
    for data in batch:
        num_input = []
        morpheme_input = []
        attention_mask = []
        words = data.split()
        # pad to fix length
        words = padding(words, config.sequence_length, '<PAD>', add_first='<BOS>', add_last='<EOS>')
        for w in words:
            if w in word2id:
                num_input.append(word2id[w])
            else:
                num_input.append(word2id['<UNK>'])

            # set morpheme input
            w_morpheme = []
            w_att_mask = []
            for idx in range(config.max_ngram):
                if is_reserved_token(w):
                    w_morpheme.append(word2id[w])
                    w_att_mask.append(-1)
                else:
                    if idx < len(w):
                        if w[idx] in word2id:
                            w_morpheme.append(word2id[w[idx]])
                            w_att_mask.append(1)
                        else:
                            w_morpheme.append(word2id['<UNK>'])
                            w_att_mask.append(1)
                    else:
                        w_morpheme.append(word2id['<MORPHEME>'])
                        w_att_mask.append(-1)
            morpheme_input.append(w_morpheme)
            attention_mask.append(w_att_mask)

        batch_input.append(num_input)
        batch_morpheme.append(morpheme_input)
        batch_attention_mask.append(attention_mask)

        #print("words:", words)
        #print("num_input:", num_input)
        #print("morpheme_input:", morpheme_input)
        #print("attention_mask:", attention_mask)

    tensor_input = torch.tensor(batch_input, dtype=torch.long).to(device)
    tensor_morpheme = torch.tensor(batch_morpheme, dtype=torch.long).to(device)
    tensor_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long).to(device)

    return tensor_input, tensor_morpheme, tensor_attention_mask

def predict(query):
    outputs = []
    input_, morpheme, attention_mask = generate_input(query)
    with torch.no_grad():
        output, attention = model(input_, morpheme, attention_mask)
        for idx in output.argmax(1).tolist():
            outputs.append(id2label[idx])

    return outputs, attention

if __name__ == "__main__":
    query = ["对呀 什么 事", "不 需要 了 不 需要 了", "不 可以"]
    output, attention = predict(query)

    for i in range(len(output)):
        print("query:", query[i], "label:", output[i], "attention:", attention[i, :, : , : ])
