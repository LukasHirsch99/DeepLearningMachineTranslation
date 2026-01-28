import torch
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
  def __init__(self, source_sentences, target_sentences, tokenizer, max_length=100):
    self.source_sentences = source_sentences
    self.target_sentences = target_sentences
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.pad_id = tokenizer.pad_id
    self.sos_id = tokenizer.sos_id
    self.eos_id = tokenizer.eos_id

  def __len__(self):
    return len(self.source_sentences)

  def __getitem__(self, idx):
    src_tokens = self.tokenizer.encode(
        self.source_sentences[idx],
        add_special_tokens=False
    )

    tgt_tokens = self.tokenizer.encode(
        self.target_sentences[idx],
        add_special_tokens=False
    )

    max_len = self.max_length - 1

    src_tokens = src_tokens[:max_len]
    tgt_tokens = tgt_tokens[:max_len]

    src_final = torch.tensor(src_tokens)

    tgt_input = torch.cat([
        torch.tensor([self.sos_id]),
        torch.tensor(tgt_tokens)
    ])

    tgt_output = torch.cat([
        torch.tensor(tgt_tokens),
        torch.tensor([self.eos_id])
    ])

    assert torch.equal(tgt_input[1:], tgt_output[:-1]), \
        "f"

    return {
        'src_tokens': src_final,
        'tgt_input': tgt_input,
        'tgt_output': tgt_output,
        'src_len': len(src_final)
    }

def collate_fn(batch, pad_idx, max_length=100):
  batch_size = len(batch)

  src_max_len = min(max([item['src_len'] for item in batch]), max_length)
  tgt_max_len = min(max([len(item['tgt_input']) for item in batch]), max_length)

  src_tokens = torch.full((batch_size, src_max_len), pad_idx, dtype=torch.long)
  tgt_input = torch.full((batch_size, tgt_max_len), pad_idx, dtype=torch.long)
  tgt_output = torch.full((batch_size, tgt_max_len), pad_idx, dtype=torch.long)
  src_lens = torch.zeros(batch_size, dtype=torch.long)
  tgt_lens = torch.zeros(batch_size, dtype=torch.long)

  for i, item in enumerate(batch):
      src_len = min(item['src_len'], src_max_len)
      tgt_len = min(len(item['tgt_input']), tgt_max_len)

      src_tokens[i, :src_len] = item['src_tokens'][:src_len]
      tgt_input[i, :tgt_len] = item['tgt_input'][:tgt_len]
      tgt_output[i, :tgt_len] = item['tgt_output'][:tgt_len]

      src_lens[i] = src_len
      tgt_lens[i] = tgt_len

  src_lens, sorted_idx = src_lens.sort(descending=True)
  src_tokens = src_tokens[sorted_idx]
  tgt_input = tgt_input[sorted_idx]
  tgt_output = tgt_output[sorted_idx]
  tgt_lens = tgt_lens[sorted_idx]

  return {
      'src_tokens': src_tokens,
      'tgt_input': tgt_input,
      'tgt_output': tgt_output,
      'src_lens': src_lens,
      'tgt_lens': tgt_lens
  }

def create_dataloader(dataset, batch_size, pad_idx, max_length=100, shuffle=True):
  return DataLoader(
      dataset=dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      collate_fn=lambda batch: collate_fn(batch, pad_idx, max_length),
      num_workers=16
  )