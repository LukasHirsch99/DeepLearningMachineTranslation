import sacrebleu
from tokenizers import Tokenizer
from utils.inference import evalute_model




def bleu_evaluation(
    model,
    test_loader, 
    tokenizer: Tokenizer,
    dataset_max_sample_len, 
    device, 
    eval_iters
):
    total_bleu = 0.0
    print_enabled = True
    samples = 0

    for k, batch in enumerate(test_loader):
        if k >= eval_iters:
            break

        src, tgt, src_key_padding_mask, tgt_key_padding_mask = batch

        src = src[:, 1:].to(device)  # [batch, src_len]
        # Also shift the padding mask
        src_key_padding_mask = src_key_padding_mask[:, 1:].to(device)
        
        tgt = tgt[:, :-1].to(device)  # [batch, tgt_len]
        # Also shift the padding mask
        tgt_key_padding_mask = tgt_key_padding_mask[:, :-1].to(device)

        output = model(
            src,
            tgt,
            src_key_padding_mask,
            tgt_key_padding_mask,
        ) # [batch, tgt_len, vocab_size]
        
        output = output.reshape(-1, output.shape[-1])
        
        

        for idx, (de, en) in enumerate(zip(batch_de, batch_en)):
            sample_de = tokenizer.decode_to_text(de.tolist())
            sample_en = tokenizer.decode_to_text(en.tolist())

            BLEUscore = sacrebleu.corpus_bleu([translation], [[sample_en]])
            total_bleu += BLEUscore.score

            if print_enabled:
                print(f"Original German: {sample_de}")
                print(f"Model Translation: '{translation}'")
                print(f"Reference Translation: {sample_en}")
                print(f"BLEU Score: {BLEUscore.score:.4f}")

    print(f"\nAverage BLEU Score over {samples} samples: {(total_bleu / samples):.4f}")
