from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def yeild_token(df, tokenizer):
    for idx, row in df.iterrows():
        yield tokenizer(row['normalized_sentence'])
        
def build_tokenizer_vocab(df, vocab_size=10000):
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(
        yeild_token(df, tokenizer),
        specials=["<unk>"],
        max_tokens=vocab_size
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab, tokenizer
