import sys
from model import TextCNN, Config
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
import time
from tools.data_loader import build_vocab, build_label
from tools.data_loader import load_vocab
from tools.data_loader import TextDataSet
from torch.utils.tensorboard import SummaryWriter
import os, glob, shutil, json
from pytorch_model_summary import summary

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--config', type=str, default="model.config")
args = parser.parse_args()

config = Config(json.load(open(args.config)))

torch.manual_seed(config.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# clear tensorboard logs
if os.path.exists(config.tensorboard):
    shutil.rmtree(config.tensorboard)
os.mkdir(config.tensorboard)
writer = SummaryWriter(log_dir=config.tensorboard, flush_secs=int(config.tensorboard_flush_sec))

checkpoint_path = "checkpoints"
if os.path.exists(checkpoint_path):
    shutil.rmtree(checkpoint_path)
os.mkdir(checkpoint_path)

# do text parsing, get vocab size and class count
build_vocab(config.train, config.label_save, config.vocab_save)
label2id, id2label = load_vocab(config.label_save)
word2id, id2word = load_vocab(config.vocab_save)

extend_config = {
    "vocab_size": len(word2id),
    "labels": len(label2id)
}
config.set(extend_config)
print(config)

# set model
model = TextCNN(config)
model.build()
model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=config.lr, eps=config.eps, weight_decay=config.weight_decay)

writer.add_graph(model, torch.randint(low=0,high=100, size=(config.batch_size, config.sequence_length), dtype=torch.long).to(device))
print(summary(model, torch.randint(low=0,high=100, size=(config.batch_size, config.sequence_length), dtype=torch.long).to(device)))

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

def generate_batch(batch):
    # TextDataSet yield one line contain label and input
    batch_label, batch_input = [], []
    for data in batch:
        num_input = []
        label, sentence = data.split('\t')
        batch_label.append(label2id[label])
        words = sentence.split()
        # pad to fix length
        words = padding(words, config.sequence_length, '<PAD>', add_first='<BOS>', add_last='<EOS>')
        for w in words:
            if w in word2id:
                num_input.append(word2id[w])
            else:
                num_input.append(word2id['<UNK>'])
        batch_input.append(num_input)

    tensor_label = torch.tensor(batch_label, dtype=torch.long)
    tensor_input = torch.tensor(batch_input, dtype=torch.long)

    return tensor_label.to(device), tensor_input.to(device)

def save_checkpoint(state, is_best=True, filename="checkpoint"):
    name = "%s_epoch:%s_steps:%s_validacc:%s.pt" % (filename, state['epoch'], state['steps'], state['valid_acc'])
    torch.save(state, "%s/%s" % (checkpoint_path, name))
    if is_best:
        shutil.copyfile("%s/%s" % (checkpoint_path, name), "%s" % (config.save_model))

def test(test_data):
    valid_loss = 0
    valid_acc = 0
    test_dataset = TextDataSet(test_data)
    data = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=generate_batch)
    model.eval()
    for i, (label, input_ids) in enumerate(data):
        with torch.no_grad():
            output = model(input_ids)
            loss = criterion(output, label)
            valid_loss += loss.item()
            valid_acc += (output.argmax(1) == label).sum().item()

    return valid_loss / len(test_dataset), valid_acc / len(test_dataset)

def train():
    global model

    best_valid_acc = 0.0
    model.train()
    steps = 0
    for epoch in range(config.epoch):
        start_time = time.time()
        train_loss = 0
        train_count = 0
        train_acc = 0
        train_dataset = TextDataSet(config.train)
        data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=generate_batch)
        for i, (label, input_ids) in enumerate(data):
            optimizer.zero_grad()
            output = model(input_ids)
            loss = criterion(output, label)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_acc += (output.argmax(1) == label).sum().item()
            train_count += input_ids.shape[0]
            step_accuracy = train_acc / train_count

            if steps % config.save_step == 0 and steps > 0:
                #print("Epoch:%s Steps:%s train_count:%s" % (epoch, steps, train_count))
                valid_loss, valid_acc = test(config.dev)
                if valid_acc > best_valid_acc and epoch > 0:
                    save_checkpoint({
                        "epoch": epoch + 1,
                        "steps": steps,
                        "state_dict": model.state_dict(),
                        "valid_acc": valid_acc
                        })
                    best_valid_acc = valid_acc

                secs = int(time.time() - start_time)
                mins = secs / 60
                secs = mins % 60
                writer.add_scalars("StepLoss", {
                    'train': train_loss / train_count,
                    "valid": valid_loss
                    }, steps)
                writer.add_scalars("StepAcc", {
                    'train': train_acc / train_count,
                    "valid": valid_acc
                    }, steps)

                print("Epoch: %d" % (epoch + 1), "Steps: %d" % steps, " | time in %d minutes, %d seconds" % (mins, secs))
                print(f"\tLoss: {train_loss / train_count:.4f}(train)\t|\tAcc: {train_acc / train_count * 100:.1f}%(train)")
                print(f"\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)")
            steps += 1

        # tensorboard for epoch accuracy and loss
        valid_loss, valid_acc = test(config.dev)
        writer.add_scalars("EpochLoss", {
            'train': train_loss / len(train_dataset),
            "valid": valid_loss
            }, epoch + 1)
        writer.add_scalars("EpochAcc", {
            'train': train_acc / len(train_dataset),
            "valid": valid_acc
            }, epoch + 1)

    # test
    model = TextCNN(config)
    model.build()
    model.to(device)
    saved_model = torch.load(config.save_model)
    model.load_state_dict(saved_model["state_dict"])
    print("epoch:%s steps:%s best_valid_acc:%s" % (saved_model["epoch"], saved_model["steps"], saved_model["valid_acc"]))

    test_loss, test_acc = test(config.test)
    print(f"\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)")


if __name__ == "__main__":
    if args.mode == 'train':
        train()
