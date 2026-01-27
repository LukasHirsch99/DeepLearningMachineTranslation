from tokenizers import Tokenizer

class BPETokenizer:
  def __init__(self, tokenizer_path="/content/drive/MyDrive/DeepLearningMachineTranslation/tokenizer.json"):
    self.tokenizer = Tokenizer.from_file(tokenizer_path)
    self.pad_token = "[PAD]"
    self.sos_token = "[SOS]"
    self.eos_token = "[EOS]"
    self.unk_token = "[UNK]"

    self.pad_id = self.tokenizer.token_to_id(self.pad_token)
    self.sos_id = self.tokenizer.token_to_id(self.sos_token)
    self.eos_id = self.tokenizer.token_to_id(self.eos_token)
    self.unk_id = self.tokenizer.token_to_id(self.unk_token)

    self.special_ids = {self.pad_id, self.sos_id, self.eos_id}

  def encode(self, text, add_special_tokens=True):
    encoding = self.tokenizer.encode(text)
    tokens = encoding.ids

    if add_special_tokens:
        tokens = [self.sos_id] + tokens + [self.eos_id]
    return tokens

  def decode(self, token_ids, skip_special_tokens=True):
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()
    if skip_special_tokens:
        token_ids = [t for t in token_ids if t not in self.special_ids]

    text = self.tokenizer.decode(token_ids)

    text = text.replace(' ##', '')

    return text

  def get_vocab_size(self):
    return self.tokenizer.get_vocab_size()

  def tokenize(self, text):
    encoding = self.tokenizer.encode(text)
    return encoding.tokens

  def __call__(self, text, add_special_tokens=True):
    return self.encode(text, add_special_tokens)