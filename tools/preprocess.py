from tokenizer import Tokenizer
import argparse

def is_punctuation(s):
    ascii_punc = "!\"#$%&'()*+,-./:;<=>?`~[]\\^_~{|}"
    # cjk punctuation
    if s in ascii_punc or (s >= "\u3000" and s <= "\u303f") or (s >= "\uff00" and s <= "\uffef") or (s >= "\ufe30" and s <= "\ufe4f"):
        return True
    return False

def remove_punctuation(text):
    res = ""
    for ch in text:
        if not is_punctuation(ch):
            res += ch

    return res

def parse_data(data, word_tokenize):
    info = data.split('\t')
    text = remove_punctuation(info[1])
    token = word_tokenize(text)

    return "%s\t%s" % (info[0], ' '.join(token))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tokenize data")
    parser.add_argument('--input', type=str, required=True, help='input file')
    parser.add_argument('--output', type=str, required=True, help='output file')
    args = parser.parse_args()

    with open(args.input) as rf:
        with open(args.output, "w+") as wf:
            tokenizer = Tokenizer().tokenize
            for line in rf:
                out = parse_data(line.strip(), tokenizer)
                wf.write("%s\n" % out)
