import os
import sys
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, Sequence
from collections import Counter

corpus = [
    "WordPiece tokenization is used in modern language models",
    "Tokenizers break words into subwords",
    "Rare words are split into smaller meaningful pieces",
    "Transformers rely on subword tokenization",
    "Misspellings like tokenizerr still work well"
]

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = Sequence([Lowercase()])
tokenizer.pre_tokenizer = Whitespace()

trainer = WordPieceTrainer(
    vocab_size=50,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

try:
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    print("Vocabulary Size:", vocab_size, flush=True)
except Exception as e:
    import traceback
    err = traceback.format_exc()
    print(err, file=sys.stderr, flush=True)
    with open(os.path.join(os.path.dirname(__file__), "error.txt"), "w") as f:
        f.write(err)
    sys.exit(1)
