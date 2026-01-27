## Download Dataset for vocab building

```bash
pip3 install datasets tokenizers torch matplotlib
```

```bash
curl -L -o ./datasets/wmt-2014-english-german.zip \
  https://www.kaggle.com/api/v1/datasets/download/mohamedlotfy50/wmt-2014-english-german
```

```bash
mkdir datasets
unzip datasets/wmt-2014-english-german.zip -d datasets/
```


### Attention is all you need base model

**Transformer model parameters**
- d_model=256
- nhead=8
- num_encoder_layers=4
- num_decoder_layers=4
- dim_feedforward=1024
- dropout=0.1
- max_len=150


**Training Hyperparameters**

- training_samples = 100_000
- batch_size = 64
- dataset_max_sample_len = 100
- num_epochs = 100
- warmup_steps = 4000
- eval_iters = 30
- label_smoothing = 0.1

**Optimizer**
- start_lr = 3e-4
- betas = (0.9, 0.98)
- epsilon = 1e-9

**Results**
- Validation Loss on test data: `3.31821`
- Bleu of `13` on test data.
