## Download Dataset for vocab building

```bash
pip3 install datasets tokenizers torch matplotlib
```

```bash
curl -L -o ./datasets/wmt-2014-english-german.zip \
  https://www.kaggle.com/api/v1/datasets/download/mohamedlotfy50/wmt-2014-english-german
```

```bash
unzip datasets/wmt-2014-english-german.zip -d datasets/
```

### V1

training_samples = 100_000
batch_size = 64
dataset_max_sample_len = 100

# Transformer model parameters
d_model=256
nhead=8
num_encoder_layers=4
num_decoder_layers=4
dim_feedforward=1024
dropout=0.0
max_len=150

# training
num_epochs = 100
warmup_steps = 2000
eval_iters = 30
patience = 30

label_smoothing = 0.1

# optimizer
start_lr = 3e-4
betas = (0.9, 0.98)
epsilon = 1e-9

No compilation

```py
Epoch 1/100
  [Batch 50/1398] - Training Loss: 9.6801
  [Batch 100/1398] - Training Loss: 9.3494
  [Batch 150/1398] - Training Loss: 9.0403
  [Batch 200/1398] - Training Loss: 8.7457
  [Batch 250/1398] - Training Loss: 8.4867
  [Batch 300/1398] - Training Loss: 8.2659
  [Batch 350/1398] - Training Loss: 8.0818
  [Batch 400/1398] - Training Loss: 7.9272
  [Batch 450/1398] - Training Loss: 7.7935
  [Batch 500/1398] - Training Loss: 7.6771
  [Batch 550/1398] - Training Loss: 7.5724
  [Batch 600/1398] - Training Loss: 7.4784
  [Batch 650/1398] - Training Loss: 7.3902
  [Batch 700/1398] - Training Loss: 7.3093
  [Batch 750/1398] - Training Loss: 7.2347
  [Batch 800/1398] - Training Loss: 7.1638
  [Batch 850/1398] - Training Loss: 7.0978
  [Batch 900/1398] - Training Loss: 7.0362
  [Batch 950/1398] - Training Loss: 6.9769
  [Batch 1000/1398] - Training Loss: 6.9221
  [Batch 1050/1398] - Training Loss: 6.8693
  [Batch 1100/1398] - Training Loss: 6.8192
  [Batch 1150/1398] - Training Loss: 6.7711
  [Batch 1200/1398] - Training Loss: 6.7248
  [Batch 1250/1398] - Training Loss: 6.6808
  [Batch 1300/1398] - Training Loss: 6.6374
  [Batch 1350/1398] - Training Loss: 6.5969

  Average Loss: 6.5588
  Validation Loss: 5.5738
  Learning Rate: 0.000219
  Time: 268.04s
  Samples/sec: ~334
  Best Loss So Far: inf

============================================================
Epoch 2/100
  [Batch 50/1398] - Training Loss: 5.3549
  [Batch 100/1398] - Training Loss: 5.3303
  [Batch 150/1398] - Training Loss: 5.3168
  [Batch 200/1398] - Training Loss: 5.3016
  [Batch 250/1398] - Training Loss: 5.2857
  [Batch 300/1398] - Training Loss: 5.2708
  [Batch 350/1398] - Training Loss: 5.2526
  [Batch 400/1398] - Training Loss: 5.2357
  [Batch 450/1398] - Training Loss: 5.2155
  [Batch 500/1398] - Training Loss: 5.1995
  [Batch 550/1398] - Training Loss: 5.1807
  [Batch 600/1398] - Training Loss: 5.1637
  [Batch 650/1398] - Training Loss: 5.1460
  [Batch 700/1398] - Training Loss: 5.1306
  [Batch 750/1398] - Training Loss: 5.1124
  [Batch 800/1398] - Training Loss: 5.0953
  [Batch 850/1398] - Training Loss: 5.0784
  [Batch 900/1398] - Training Loss: 5.0615
  [Batch 950/1398] - Training Loss: 5.0444
  [Batch 1000/1398] - Training Loss: 5.0284
  [Batch 1050/1398] - Training Loss: 5.0127
  [Batch 1100/1398] - Training Loss: 4.9950
  [Batch 1150/1398] - Training Loss: 4.9795
  [Batch 1200/1398] - Training Loss: 4.9638
  [Batch 1250/1398] - Training Loss: 4.9473
  [Batch 1300/1398] - Training Loss: 4.9321
  [Batch 1350/1398] - Training Loss: 4.9157

  Average Loss: 4.9010
  Validation Loss: 4.6372
  Learning Rate: 0.000300
  Time: 267.09s
  Samples/sec: ~335
  Best Loss So Far: 5.5738

============================================================
Epoch 3/100
  [Batch 50/1398] - Training Loss: 4.2744
  [Batch 100/1398] - Training Loss: 4.2720
  [Batch 150/1398] - Training Loss: 4.2589
  [Batch 200/1398] - Training Loss: 4.2492
  [Batch 250/1398] - Training Loss: 4.2419
  [Batch 300/1398] - Training Loss: 4.2359
  [Batch 350/1398] - Training Loss: 4.2303
  [Batch 400/1398] - Training Loss: 4.2250
  [Batch 450/1398] - Training Loss: 4.2201
  [Batch 500/1398] - Training Loss: 4.2123
  [Batch 550/1398] - Training Loss: 4.2063
  [Batch 600/1398] - Training Loss: 4.1995
  [Batch 650/1398] - Training Loss: 4.1951
  [Batch 700/1398] - Training Loss: 4.1893
  [Batch 750/1398] - Training Loss: 4.1817
  [Batch 800/1398] - Training Loss: 4.1755
  [Batch 850/1398] - Training Loss: 4.1707
  [Batch 900/1398] - Training Loss: 4.1661
  [Batch 950/1398] - Training Loss: 4.1612
  [Batch 1000/1398] - Training Loss: 4.1563
  [Batch 1050/1398] - Training Loss: 4.1510
  [Batch 1100/1398] - Training Loss: 4.1464
  [Batch 1150/1398] - Training Loss: 4.1407
  [Batch 1200/1398] - Training Loss: 4.1351
  [Batch 1250/1398] - Training Loss: 4.1309
  [Batch 1300/1398] - Training Loss: 4.1262
  [Batch 1350/1398] - Training Loss: 4.1203

  Average Loss: 4.1152
  Validation Loss: 4.3031
  Learning Rate: 0.000300
  Time: 266.74s
  Samples/sec: ~335
  Best Loss So Far: 4.6372

=======================================================
Epoch 4/100
  [Batch 50/1398] - Training Loss: 3.7183
  [Batch 100/1398] - Training Loss: 3.6903
  [Batch 150/1398] - Training Loss: 3.6891
  [Batch 200/1398] - Training Loss: 3.6943
  [Batch 250/1398] - Training Loss: 3.6949
  [Batch 300/1398] - Training Loss: 3.6962
  [Batch 350/1398] - Training Loss: 3.6967
  [Batch 400/1398] - Training Loss: 3.6992
  [Batch 450/1398] - Training Loss: 3.7014
  [Batch 500/1398] - Training Loss: 3.7044
  [Batch 550/1398] - Training Loss: 3.7049
  [Batch 600/1398] - Training Loss: 3.7060
  [Batch 650/1398] - Training Loss: 3.7064
  [Batch 700/1398] - Training Loss: 3.7081
  [Batch 750/1398] - Training Loss: 3.7085
  [Batch 800/1398] - Training Loss: 3.7082
  [Batch 850/1398] - Training Loss: 3.7088
  [Batch 900/1398] - Training Loss: 3.7085
  [Batch 950/1398] - Training Loss: 3.7078
  [Batch 1000/1398] - Training Loss: 3.7082
  [Batch 1050/1398] - Training Loss: 3.7068
  [Batch 1100/1398] - Training Loss: 3.7065
  [Batch 1150/1398] - Training Loss: 3.7066
  [Batch 1200/1398] - Training Loss: 3.7063
  [Batch 1250/1398] - Training Loss: 3.7053
  [Batch 1300/1398] - Training Loss: 3.7044
  [Batch 1350/1398] - Training Loss: 3.7038

  Average Loss: 3.7032
  Validation Loss: 4.1174
  Learning Rate: 0.000300
  Time: 268.20s
  Samples/sec: ~334
  Best Loss So Far: 4.3031

=======================================================
Epoch 5/100
  [Batch 50/1398] - Training Loss: 3.3817
  [Batch 100/1398] - Training Loss: 3.3756
  [Batch 150/1398] - Training Loss: 3.3825
  [Batch 200/1398] - Training Loss: 3.3805
  [Batch 250/1398] - Training Loss: 3.3872
  [Batch 300/1398] - Training Loss: 3.3919
  [Batch 350/1398] - Training Loss: 3.3935
  [Batch 400/1398] - Training Loss: 3.3977
  [Batch 450/1398] - Training Loss: 3.4021
  [Batch 500/1398] - Training Loss: 3.4028
  [Batch 550/1398] - Training Loss: 3.4047
  [Batch 600/1398] - Training Loss: 3.4070
  [Batch 650/1398] - Training Loss: 3.4084
  [Batch 700/1398] - Training Loss: 3.4109
  [Batch 750/1398] - Training Loss: 3.4133
  [Batch 800/1398] - Training Loss: 3.4148
  [Batch 850/1398] - Training Loss: 3.4164
  [Batch 900/1398] - Training Loss: 3.4188
  [Batch 950/1398] - Training Loss: 3.4199
  [Batch 1000/1398] - Training Loss: 3.4231
  [Batch 1050/1398] - Training Loss: 3.4245
  [Batch 1100/1398] - Training Loss: 3.4274
  [Batch 1150/1398] - Training Loss: 3.4287
  [Batch 1200/1398] - Training Loss: 3.4295
  [Batch 1250/1398] - Training Loss: 3.4305
  [Batch 1300/1398] - Training Loss: 3.4330
  [Batch 1350/1398] - Training Loss: 3.4341

  Average Loss: 3.4355
  Validation Loss: 4.0525
  Learning Rate: 0.000300
  Time: 266.82s
  Samples/sec: ~335
  Best Loss So Far: 4.1174

=======================================================
Epoch 6/100
  [Batch 50/1398] - Training Loss: 3.1247
  [Batch 100/1398] - Training Loss: 3.1329
  [Batch 150/1398] - Training Loss: 3.1344
  [Batch 200/1398] - Training Loss: 3.1432
  [Batch 250/1398] - Training Loss: 3.1478
  [Batch 300/1398] - Training Loss: 3.1560
  [Batch 350/1398] - Training Loss: 3.1634
  [Batch 400/1398] - Training Loss: 3.1673
  [Batch 450/1398] - Training Loss: 3.1718
  [Batch 500/1398] - Training Loss: 3.1748
  [Batch 550/1398] - Training Loss: 3.1790
  [Batch 600/1398] - Training Loss: 3.1812
  [Batch 650/1398] - Training Loss: 3.1854
  [Batch 700/1398] - Training Loss: 3.1896
  [Batch 750/1398] - Training Loss: 3.1931
  [Batch 800/1398] - Training Loss: 3.1974
  [Batch 850/1398] - Training Loss: 3.2018
  [Batch 900/1398] - Training Loss: 3.2050
  [Batch 950/1398] - Training Loss: 3.2074
  [Batch 1000/1398] - Training Loss: 3.2090
  [Batch 1050/1398] - Training Loss: 3.2127
  [Batch 1100/1398] - Training Loss: 3.2150
  [Batch 1150/1398] - Training Loss: 3.2165
  [Batch 1200/1398] - Training Loss: 3.2192
  [Batch 1250/1398] - Training Loss: 3.2207
  [Batch 1300/1398] - Training Loss: 3.2232
  [Batch 1350/1398] - Training Loss: 3.2263

  Average Loss: 3.2276
  Validation Loss: 4.0536
  Learning Rate: 0.000300
  Time: 267.59s
  Samples/sec: ~334
  Best Loss So Far: 4.0525
```

### V2

training_samples = 100_000
batch_size = 64
dataset_max_sample_len = 100

# Transformer model parameters
d_model=256
nhead=8
num_encoder_layers=4
num_decoder_layers=4
dim_feedforward=1024
dropout=0.1
max_len=150

# training
num_epochs = 100
warmup_steps = 2000
eval_iters = 30
patience = 30

label_smoothing = 0.1

# optimizer
start_lr = 3e-4
betas = (0.9, 0.98)
epsilon = 1e-9

With compilation

Bleu of 22.3696 on test data.
  Average Loss: 3.1376
  Validation Loss: 3.8523
  Learning Rate: 0.000300
  Time: 271.58s
  Samples/sec: ~329
  Best Loss So Far: 3.8521

```py
Epoch 1/100
  [Batch 50/1398] - Training Loss: 9.6736
  [Batch 100/1398] - Training Loss: 9.3490
  [Batch 150/1398] - Training Loss: 9.0421
  [Batch 200/1398] - Training Loss: 8.7543
  [Batch 250/1398] - Training Loss: 8.5023
  [Batch 300/1398] - Training Loss: 8.2949
  [Batch 350/1398] - Training Loss: 8.1252
  [Batch 400/1398] - Training Loss: 7.9836
  [Batch 450/1398] - Training Loss: 7.8583
  [Batch 500/1398] - Training Loss: 7.7503
  [Batch 550/1398] - Training Loss: 7.6511
  [Batch 600/1398] - Training Loss: 7.5638
  [Batch 650/1398] - Training Loss: 7.4847
  [Batch 700/1398] - Training Loss: 7.4103
  [Batch 750/1398] - Training Loss: 7.3420
  [Batch 800/1398] - Training Loss: 7.2782
  [Batch 850/1398] - Training Loss: 7.2179
  [Batch 900/1398] - Training Loss: 7.1605
  [Batch 950/1398] - Training Loss: 7.1062
  [Batch 1000/1398] - Training Loss: 7.0551
  [Batch 1050/1398] - Training Loss: 7.0068
  [Batch 1100/1398] - Training Loss: 6.9617
  [Batch 1150/1398] - Training Loss: 6.9172
  [Batch 1200/1398] - Training Loss: 6.8749
  [Batch 1250/1398] - Training Loss: 6.8339
  [Batch 1300/1398] - Training Loss: 6.7945
  [Batch 1350/1398] - Training Loss: 6.7568

  Average Loss: 6.7217
  Validation Loss: 5.7246
  Learning Rate: 0.000219
  Time: 400.69s
  Samples/sec: ~223
  Best Loss So Far: inf
============================================================

Epoch 2/100
  [Batch 50/1398] - Training Loss: 5.6353
  [Batch 100/1398] - Training Loss: 5.6175
  [Batch 150/1398] - Training Loss: 5.6089
  [Batch 200/1398] - Training Loss: 5.5904
  [Batch 250/1398] - Training Loss: 5.5783
  [Batch 300/1398] - Training Loss: 5.5587
  [Batch 350/1398] - Training Loss: 5.5408
  [Batch 400/1398] - Training Loss: 5.5242
  [Batch 450/1398] - Training Loss: 5.5054
  [Batch 500/1398] - Training Loss: 5.4863
  [Batch 550/1398] - Training Loss: 5.4672
  [Batch 600/1398] - Training Loss: 5.4507
  [Batch 650/1398] - Training Loss: 5.4332
  [Batch 700/1398] - Training Loss: 5.4142
  [Batch 750/1398] - Training Loss: 5.3978
  [Batch 800/1398] - Training Loss: 5.3814
  [Batch 850/1398] - Training Loss: 5.3644
  [Batch 900/1398] - Training Loss: 5.3488
  [Batch 950/1398] - Training Loss: 5.3325
  [Batch 1000/1398] - Training Loss: 5.3163
  [Batch 1050/1398] - Training Loss: 5.2992
  [Batch 1100/1398] - Training Loss: 5.2838
  [Batch 1150/1398] - Training Loss: 5.2684
  [Batch 1200/1398] - Training Loss: 5.2542
  [Batch 1250/1398] - Training Loss: 5.2398
  [Batch 1300/1398] - Training Loss: 5.2248
  [Batch 1350/1398] - Training Loss: 5.2094

  Average Loss: 5.1963
  Validation Loss: 4.8714
  Learning Rate: 0.000300
  Time: 273.21s
  Samples/sec: ~327
  Best Loss So Far: 5.7246

============================================================
Epoch 3/100
  [Batch 50/1398] - Training Loss: 4.6647
  [Batch 100/1398] - Training Loss: 4.6716
  [Batch 150/1398] - Training Loss: 4.6550
  [Batch 200/1398] - Training Loss: 4.6566
  [Batch 250/1398] - Training Loss: 4.6493
  [Batch 300/1398] - Training Loss: 4.6395
  [Batch 350/1398] - Training Loss: 4.6321
  [Batch 400/1398] - Training Loss: 4.6258
  [Batch 450/1398] - Training Loss: 4.6175
  [Batch 500/1398] - Training Loss: 4.6105
  [Batch 550/1398] - Training Loss: 4.6027
  [Batch 600/1398] - Training Loss: 4.5926
  [Batch 650/1398] - Training Loss: 4.5851
  [Batch 700/1398] - Training Loss: 4.5799
  [Batch 750/1398] - Training Loss: 4.5716
  [Batch 800/1398] - Training Loss: 4.5648
  [Batch 850/1398] - Training Loss: 4.5557
  [Batch 900/1398] - Training Loss: 4.5496
  [Batch 950/1398] - Training Loss: 4.5423
  [Batch 1000/1398] - Training Loss: 4.5347
  [Batch 1050/1398] - Training Loss: 4.5280
  [Batch 1100/1398] - Training Loss: 4.5215
  [Batch 1150/1398] - Training Loss: 4.5148
  [Batch 1200/1398] - Training Loss: 4.5075
  [Batch 1250/1398] - Training Loss: 4.5007
  [Batch 1300/1398] - Training Loss: 4.4934
  [Batch 1350/1398] - Training Loss: 4.4880

  Average Loss: 4.4810
  Validation Loss: 4.3939
  Learning Rate: 0.000300
  Time: 272.46s
  Samples/sec: ~328
  Best Loss So Far: 4.8714

============================================================
Epoch 4/100
  [Batch 50/1398] - Training Loss: 4.1486
  [Batch 100/1398] - Training Loss: 4.1344
  [Batch 150/1398] - Training Loss: 4.1287
  [Batch 200/1398] - Training Loss: 4.1271
  [Batch 250/1398] - Training Loss: 4.1261
  [Batch 300/1398] - Training Loss: 4.1265
  [Batch 350/1398] - Training Loss: 4.1257
  [Batch 400/1398] - Training Loss: 4.1210
  [Batch 450/1398] - Training Loss: 4.1181
  [Batch 500/1398] - Training Loss: 4.1176
  [Batch 550/1398] - Training Loss: 4.1148
  [Batch 600/1398] - Training Loss: 4.1125
  [Batch 650/1398] - Training Loss: 4.1107
  [Batch 700/1398] - Training Loss: 4.1077
  [Batch 750/1398] - Training Loss: 4.1066
  [Batch 800/1398] - Training Loss: 4.1060
  [Batch 850/1398] - Training Loss: 4.1039
  [Batch 900/1398] - Training Loss: 4.1012
  [Batch 950/1398] - Training Loss: 4.0977
  [Batch 1000/1398] - Training Loss: 4.0965
  [Batch 1050/1398] - Training Loss: 4.0938
  [Batch 1100/1398] - Training Loss: 4.0922
  [Batch 1150/1398] - Training Loss: 4.0888
  [Batch 1200/1398] - Training Loss: 4.0868
  [Batch 1250/1398] - Training Loss: 4.0835
  [Batch 1300/1398] - Training Loss: 4.0810
  [Batch 1350/1398] - Training Loss: 4.0795

  Average Loss: 4.0778
  Validation Loss: 4.1906
  Learning Rate: 0.000300
  Time: 272.68s
  Samples/sec: ~328
  Best Loss So Far: 4.3939

============================================================
Epoch 5/100
  [Batch 50/1398] - Training Loss: 3.8264
  [Batch 100/1398] - Training Loss: 3.8241
  [Batch 150/1398] - Training Loss: 3.8309
  [Batch 200/1398] - Training Loss: 3.8332
  [Batch 250/1398] - Training Loss: 3.8353
  [Batch 300/1398] - Training Loss: 3.8328
  [Batch 350/1398] - Training Loss: 3.8362
  [Batch 400/1398] - Training Loss: 3.8363
  [Batch 450/1398] - Training Loss: 3.8376
  [Batch 500/1398] - Training Loss: 3.8375
  [Batch 550/1398] - Training Loss: 3.8372
  [Batch 600/1398] - Training Loss: 3.8394
  [Batch 650/1398] - Training Loss: 3.8397
  [Batch 700/1398] - Training Loss: 3.8394
  [Batch 750/1398] - Training Loss: 3.8389
  [Batch 800/1398] - Training Loss: 3.8378
  [Batch 850/1398] - Training Loss: 3.8370
  [Batch 900/1398] - Training Loss: 3.8360
  [Batch 950/1398] - Training Loss: 3.8359
  [Batch 1000/1398] - Training Loss: 3.8348
  [Batch 1050/1398] - Training Loss: 3.8351
  [Batch 1100/1398] - Training Loss: 3.8355
  [Batch 1150/1398] - Training Loss: 3.8352
  [Batch 1200/1398] - Training Loss: 3.8349
  [Batch 1250/1398] - Training Loss: 3.8340
  [Batch 1300/1398] - Training Loss: 3.8339
  [Batch 1350/1398] - Training Loss: 3.8326

  Average Loss: 3.8320
  Validation Loss: 4.0797
  Learning Rate: 0.000300
  Time: 273.23s
  Samples/sec: ~327
  Best Loss So Far: 4.1906

============================================================
Epoch 6/100
  [Batch 50/1398] - Training Loss: 3.6225
  [Batch 100/1398] - Training Loss: 3.6146
  [Batch 150/1398] - Training Loss: 3.6283
  [Batch 200/1398] - Training Loss: 3.6338
  [Batch 250/1398] - Training Loss: 3.6367
  [Batch 300/1398] - Training Loss: 3.6397
  [Batch 350/1398] - Training Loss: 3.6422
  [Batch 400/1398] - Training Loss: 3.6465
  [Batch 450/1398] - Training Loss: 3.6454
  [Batch 500/1398] - Training Loss: 3.6461
  [Batch 550/1398] - Training Loss: 3.6482
  [Batch 600/1398] - Training Loss: 3.6494
  [Batch 650/1398] - Training Loss: 3.6504
  [Batch 700/1398] - Training Loss: 3.6507
  [Batch 750/1398] - Training Loss: 3.6534
  [Batch 800/1398] - Training Loss: 3.6539
  [Batch 850/1398] - Training Loss: 3.6550
  [Batch 900/1398] - Training Loss: 3.6559
  [Batch 950/1398] - Training Loss: 3.6566
  [Batch 1000/1398] - Training Loss: 3.6569
  [Batch 1050/1398] - Training Loss: 3.6568
  [Batch 1100/1398] - Training Loss: 3.6579
  [Batch 1150/1398] - Training Loss: 3.6582
  [Batch 1200/1398] - Training Loss: 3.6590
  [Batch 1250/1398] - Training Loss: 3.6589
  [Batch 1300/1398] - Training Loss: 3.6596
  [Batch 1350/1398] - Training Loss: 3.6603

  Average Loss: 3.6600
  Validation Loss: 3.9584
  Learning Rate: 0.000300
  Time: 271.98s
  Samples/sec: ~329
  Best Loss So Far: 4.0797

============================================================
Epoch 7/100
  [Batch 50/1398] - Training Loss: 3.4698
  [Batch 100/1398] - Training Loss: 3.4773
  [Batch 150/1398] - Training Loss: 3.4771
  [Batch 200/1398] - Training Loss: 3.4821
  [Batch 250/1398] - Training Loss: 3.4878
  [Batch 300/1398] - Training Loss: 3.4944
  [Batch 350/1398] - Training Loss: 3.4971
  [Batch 400/1398] - Training Loss: 3.4979
  [Batch 450/1398] - Training Loss: 3.5019
  [Batch 500/1398] - Training Loss: 3.5052
  [Batch 550/1398] - Training Loss: 3.5111
  [Batch 600/1398] - Training Loss: 3.5116
  [Batch 650/1398] - Training Loss: 3.5138
  [Batch 700/1398] - Training Loss: 3.5134
  [Batch 750/1398] - Training Loss: 3.5145
  [Batch 800/1398] - Training Loss: 3.5160
  [Batch 850/1398] - Training Loss: 3.5170
  [Batch 900/1398] - Training Loss: 3.5181
  [Batch 950/1398] - Training Loss: 3.5201
  [Batch 1000/1398] - Training Loss: 3.5216
  [Batch 1050/1398] - Training Loss: 3.5226
  [Batch 1100/1398] - Training Loss: 3.5236
  [Batch 1150/1398] - Training Loss: 3.5245
  [Batch 1200/1398] - Training Loss: 3.5258
  [Batch 1250/1398] - Training Loss: 3.5267
  [Batch 1300/1398] - Training Loss: 3.5276
  [Batch 1350/1398] - Training Loss: 3.5293

  Average Loss: 3.5305
  Validation Loss: 3.9301
  Learning Rate: 0.000300
  Time: 271.82s
  Samples/sec: ~329
  Best Loss So Far: 3.9584

============================================================
Epoch 8/100
  [Batch 50/1398] - Training Loss: 3.3736
  [Batch 100/1398] - Training Loss: 3.3640
  [Batch 150/1398] - Training Loss: 3.3740
  [Batch 200/1398] - Training Loss: 3.3762
  [Batch 250/1398] - Training Loss: 3.3762
  [Batch 300/1398] - Training Loss: 3.3765
  [Batch 350/1398] - Training Loss: 3.3812
  [Batch 400/1398] - Training Loss: 3.3850
  [Batch 450/1398] - Training Loss: 3.3864
  [Batch 500/1398] - Training Loss: 3.3902
  [Batch 550/1398] - Training Loss: 3.3945
  [Batch 600/1398] - Training Loss: 3.3977
  [Batch 650/1398] - Training Loss: 3.3994
  [Batch 700/1398] - Training Loss: 3.4016
  [Batch 750/1398] - Training Loss: 3.4048
  [Batch 800/1398] - Training Loss: 3.4058
  [Batch 850/1398] - Training Loss: 3.4071
  [Batch 900/1398] - Training Loss: 3.4097
  [Batch 950/1398] - Training Loss: 3.4115
  [Batch 1000/1398] - Training Loss: 3.4120
  [Batch 1050/1398] - Training Loss: 3.4133
  [Batch 1100/1398] - Training Loss: 3.4151
  [Batch 1150/1398] - Training Loss: 3.4154
  [Batch 1200/1398] - Training Loss: 3.4174
  [Batch 1250/1398] - Training Loss: 3.4188
  [Batch 1300/1398] - Training Loss: 3.4202
  [Batch 1350/1398] - Training Loss: 3.4214

  Average Loss: 3.4236
  Validation Loss: 3.8984
  Learning Rate: 0.000300
  Time: 271.80s
  Samples/sec: ~329
  Best Loss So Far: 3.9301

============================================================
Epoch 9/100
  [Batch 50/1398] - Training Loss: 3.2494
  [Batch 100/1398] - Training Loss: 3.2640
  [Batch 150/1398] - Training Loss: 3.2644
  [Batch 200/1398] - Training Loss: 3.2704
  [Batch 250/1398] - Training Loss: 3.2718
  [Batch 300/1398] - Training Loss: 3.2792
  [Batch 350/1398] - Training Loss: 3.2808
  [Batch 400/1398] - Training Loss: 3.2857
  [Batch 450/1398] - Training Loss: 3.2887
  [Batch 500/1398] - Training Loss: 3.2920
  [Batch 550/1398] - Training Loss: 3.2957
  [Batch 600/1398] - Training Loss: 3.2990
  [Batch 650/1398] - Training Loss: 3.3014
  [Batch 700/1398] - Training Loss: 3.3056
  [Batch 750/1398] - Training Loss: 3.3100
  [Batch 800/1398] - Training Loss: 3.3128
  [Batch 850/1398] - Training Loss: 3.3153
  [Batch 900/1398] - Training Loss: 3.3186
  [Batch 950/1398] - Training Loss: 3.3205
  [Batch 1000/1398] - Training Loss: 3.3223
  [Batch 1050/1398] - Training Loss: 3.3239
  [Batch 1100/1398] - Training Loss: 3.3250
  [Batch 1150/1398] - Training Loss: 3.3268
  [Batch 1200/1398] - Training Loss: 3.3288
  [Batch 1250/1398] - Training Loss: 3.3300
  [Batch 1300/1398] - Training Loss: 3.3326
  [Batch 1350/1398] - Training Loss: 3.3346

  Average Loss: 3.3358
  Validation Loss: 3.8521
  Learning Rate: 0.000300
  Time: 272.91s
  Samples/sec: ~328
  Best Loss So Far: 3.8984

============================================================
Epoch 10/100
  [Batch 50/1398] - Training Loss: 3.1849
  [Batch 100/1398] - Training Loss: 3.1803
  [Batch 150/1398] - Training Loss: 3.1907
  [Batch 200/1398] - Training Loss: 3.1916
  [Batch 250/1398] - Training Loss: 3.1968
  [Batch 300/1398] - Training Loss: 3.2008
  [Batch 350/1398] - Training Loss: 3.2057
  [Batch 400/1398] - Training Loss: 3.2083
  [Batch 450/1398] - Training Loss: 3.2122
  [Batch 500/1398] - Training Loss: 3.2161
  [Batch 550/1398] - Training Loss: 3.2184
  [Batch 600/1398] - Training Loss: 3.2224
  [Batch 650/1398] - Training Loss: 3.2257
  [Batch 700/1398] - Training Loss: 3.2289
  [Batch 750/1398] - Training Loss: 3.2322
  [Batch 800/1398] - Training Loss: 3.2349
  [Batch 850/1398] - Training Loss: 3.2371
  [Batch 900/1398] - Training Loss: 3.2394
  [Batch 950/1398] - Training Loss: 3.2421
  [Batch 1000/1398] - Training Loss: 3.2436
  [Batch 1050/1398] - Training Loss: 3.2466
  [Batch 1100/1398] - Training Loss: 3.2497
  [Batch 1150/1398] - Training Loss: 3.2524
  [Batch 1200/1398] - Training Loss: 3.2544
  [Batch 1250/1398] - Training Loss: 3.2555
  [Batch 1300/1398] - Training Loss: 3.2572
  [Batch 1350/1398] - Training Loss: 3.2587

  Average Loss: 3.2608
  Validation Loss: 3.8685
  Learning Rate: 0.000300
  Time: 272.49s
  Samples/sec: ~328
  Best Loss So Far: 3.8521

============================================================
Epoch 11/100
  [Batch 50/1398] - Training Loss: 3.1184
  [Batch 100/1398] - Training Loss: 3.1237
  [Batch 150/1398] - Training Loss: 3.1271
  [Batch 200/1398] - Training Loss: 3.1299
  [Batch 250/1398] - Training Loss: 3.1308
  [Batch 300/1398] - Training Loss: 3.1342
  [Batch 350/1398] - Training Loss: 3.1361
  [Batch 400/1398] - Training Loss: 3.1398
  [Batch 450/1398] - Training Loss: 3.1437
  [Batch 500/1398] - Training Loss: 3.1493
  [Batch 550/1398] - Training Loss: 3.1527
  [Batch 600/1398] - Training Loss: 3.1557
  [Batch 650/1398] - Training Loss: 3.1583
  [Batch 700/1398] - Training Loss: 3.1631
  [Batch 750/1398] - Training Loss: 3.1662
  [Batch 800/1398] - Training Loss: 3.1690
  [Batch 850/1398] - Training Loss: 3.1711
  [Batch 900/1398] - Training Loss: 3.1726
  [Batch 950/1398] - Training Loss: 3.1749
  [Batch 1000/1398] - Training Loss: 3.1789
  [Batch 1050/1398] - Training Loss: 3.1809
  [Batch 1100/1398] - Training Loss: 3.1825
  [Batch 1150/1398] - Training Loss: 3.1852
  [Batch 1200/1398] - Training Loss: 3.1876
  [Batch 1250/1398] - Training Loss: 3.1903
  [Batch 1300/1398] - Training Loss: 3.1917
  [Batch 1350/1398] - Training Loss: 3.1933

  Average Loss: 3.1952
  Validation Loss: 3.8598
  Learning Rate: 0.000300
  Time: 271.75s
  Samples/sec: ~329
  Best Loss So Far: 3.8521

============================================================
Epoch 12/100
  [Batch 50/1398] - Training Loss: 3.0514
  [Batch 100/1398] - Training Loss: 3.0583
  [Batch 150/1398] - Training Loss: 3.0627
  [Batch 200/1398] - Training Loss: 3.0702
  [Batch 250/1398] - Training Loss: 3.0714
  [Batch 300/1398] - Training Loss: 3.0762
  [Batch 350/1398] - Training Loss: 3.0811
  [Batch 400/1398] - Training Loss: 3.0817
  [Batch 450/1398] - Training Loss: 3.0843
  [Batch 500/1398] - Training Loss: 3.0882
  [Batch 550/1398] - Training Loss: 3.0921
  [Batch 600/1398] - Training Loss: 3.0951
  [Batch 650/1398] - Training Loss: 3.0998
  [Batch 700/1398] - Training Loss: 3.1032
  [Batch 750/1398] - Training Loss: 3.1062
  [Batch 800/1398] - Training Loss: 3.1096
  [Batch 850/1398] - Training Loss: 3.1116
  [Batch 900/1398] - Training Loss: 3.1134
  [Batch 950/1398] - Training Loss: 3.1166
  [Batch 1000/1398] - Training Loss: 3.1203
  [Batch 1050/1398] - Training Loss: 3.1224
  [Batch 1100/1398] - Training Loss: 3.1242
  [Batch 1150/1398] - Training Loss: 3.1263
  [Batch 1200/1398] - Training Loss: 3.1287
  [Batch 1250/1398] - Training Loss: 3.1308
  [Batch 1300/1398] - Training Loss: 3.1336
  [Batch 1350/1398] - Training Loss: 3.1363

  Average Loss: 3.1376
  Validation Loss: 3.8523
  Learning Rate: 0.000300
  Time: 271.58s
  Samples/sec: ~329
  Best Loss So Far: 3.8521
```
