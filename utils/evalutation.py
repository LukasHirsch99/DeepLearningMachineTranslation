import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import time

def evaluate_model(model, dataloader, tokenizer, max_len=50, device='cpu', show_examples=3):
  """
  Evaluate GRUSeq2Seq model with BLEU scores
  """
  model.eval()
  smoothing = SmoothingFunction().method4
  
  print("EVALUTATION")

  start_time = time.time()
  all_predictions = []
  all_references = []
  example_data = []
  
  print("Prozessing batches takes a while")
  with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader):
      src = batch['src_tokens'].to(device)
      src_lens = batch['src_lens'].to(device)
      trg = batch['tgt_input'].to(device)
      trg_lens = batch.get('tgt_lens', None)
      
      predictions = model.inference(src, src_lens, max_len=max_len)
      
      for i in range(src.size(0)):
        pred_tokens = predictions[i]
        pred_text = tokenizer.decode(pred_tokens.cpu().numpy(), skip_special_tokens=True)
        
        if 'tgt_output' in batch:
          ref_tokens = batch['tgt_output'][i]
        else:
          ref_tokens = trg[i]
        
        ref_text = tokenizer.decode(ref_tokens.cpu().numpy(), skip_special_tokens=True)
        
        pred_words = pred_text.split()
        ref_words = ref_text.split()
        
        all_predictions.append(pred_words)
        all_references.append([ref_words])
        
        if len(example_data) < show_examples:
          src_text = tokenizer.decode(src[i].cpu().numpy(), skip_special_tokens=True)
          example_data.append({
            'source': src_text,
            'prediction': pred_text,
            'reference': ref_text
          })
  
  bleu_scores = {}
  
  for n in range(1, 5):
    weights = tuple([1.0/n] * n + [0.0] * (4-n))
    try:
      score = corpus_bleu(
          all_references,
          all_predictions,
          weights=weights,
          smoothing_function=smoothing
      )
      bleu_scores[f'BLEU-{n}'] = score * 100
    except:
      bleu_scores[f'BLEU-{n}'] = 0.0
  
  try:
    bleu_scores['BLEU'] = corpus_bleu(
        all_references,
        all_predictions,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing
    ) * 100
  except:
    bleu_scores['BLEU'] = 0.0

  print(f"\nExample translations from test set:")
  for i, example in enumerate(example_data):
    print(f"\nExample {i+1}:")
    print(f"Source:     {example['source'][:100]}...")
    print(f"Prediction: {example['prediction'][:100]}...")
    print(f"Reference:  {example['reference'][:100]}...")
  
  print(f"\nEvaluation completed in {time.time() - start_time:.2f}s")
  print(f"Total samples: {len(all_predictions)}")
  print(f"\nBLEU Scores:")
  for name, score in bleu_scores.items():
    print(f"  {name:8}: {score:6.2f}")
  
  model.train()
  return bleu_scores
