# 第23回　アルゴリズムコンテスト
テーマ：三文字の崩し文字認識

## 概要
- 公式HP: https://sites.google.com/view/alcon2019/%E3%83%9B%E3%83%BC%E3%83%A0?authuser=0
- 提出場所: https://competitions.codalab.org/competitions/20388?secret_key=88458b35-8673-44c2-9bd6-17586adcf5a1
- 期間: 2019/05/31 ~ 2019/08/31

## 手順
1. データ詳細テーブルと変換器を生成する。
```
$ cd src
$ python creat_vocab.py # make vocabrary
$ python creat_table_multiprocess.py # make meta files
$ cd ../
```
2. 学習/推論をする。
```
$ cd experiment
$ vim ../param/expN.yaml # change params
$ python expN.py
```

3. アンサンブル

4. 提出(test_prediction.csvにする必要あり)
```
$ zip yoursubmission.zip test_prediction.csv
updating: test_prediction.csv (deflated 79%)
$ unzip -l yoursubmission.zip
Archive:  yoursubmission.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
   357417  03-22-2019 10:17   test_prediction.csv
---------                     -------
   357417                     1 file
```

## Result

### Public Result

| exp  | Public score |
| :--: | ------------ |
|  7   | 94.7000%     |
|  8   |              |
|  9   |              |
|  10  |              |
|  11  |              |





### Local Result

---

| exp No. | Local CV | fold0  | Public  | model   | resolution | comment |
| ------: | -------: | -----: | ------: | :------ | ------:    | :------ |
| 0       |  87.391% |        |         |         |            | test example. only fold0. |
| 1 (2019-06-13_10-46-46) |          | 87.301% |         | ResNet18 | 1  | MultiStepLR,momentumSGD |
| 2       |          |        |         |         |            |         |
| 3       |          |        |         |         |            |         |
| 4.1 (2019-06-20_01-08-53) |          | 90.695% |         | ResNet50+LSTM(bi-directional) | 2 | MultiStepLR, momentumSGD,CutOut 120x120|
| 4.2 (2019-06-23_01-57-06) |          | 92.302% |         | ResNet50+LSTM(bi-directional)  | 2|  MultiStepLR, momentumSGD,CutOut height//2 x width//2      |
| 6 (2019-06-24_04-20-17) |          | 85.695% |         | ResNet50+bi-LSTM+ABN | 2 | MultiStepLR, momentumSGD,CutOut height//2 x width//2  |
| 4.3 (2019-06-25_17-50-21) |          | 92.556%  |         | ResNet50+bi-GRU  | 2 | MultiStepLR, momentumSGD,CutOut height//2 x width//2 |
| 4.4 (2019-06-27_00-17-53)    |          | 88.765% |         | ResNetResLSTM_MLP |   2 | MultiStepLR, momentumSGD,CutOut height//2 x width//2 |
| 4.5 (2019-06-28_02-06-42)    |          | 92.943% |         | ResNetGRU2 |   2 | MultiStepLR, momentumSGD,CutOut height//2 x width//2. 学習不十分かもしれない |
| 4.6 (2019-06-30_07-33-17) |           | 92.710% |        | ResNetGRU3 |  2 | Grad clip(1.0) |
| 4.7 (2019-07-01_15-10-22) |   | 94.062%  |  | OctResNetGRU2 | 6 |   |
| 7.1 (2019-07-07_06-58-19) |    | 98.267% |   | OctResNetGRU2 | 6 | SSE(5epoch/cycle) CEL |

### EXP-7

---

#### Detail

- Model : OctConv ResNet50 + BiGRU x 2
- Batch Size: 16 (about 6000 iter / epoch)
- Resolution: 6 (192 x 128  <--- 210 x 150)
- FP32
- SGDR(5epoch / cycle)
- 3~7cycle (5cycle) で SnapShot Ensemble
- total epoch: 35

#### Result

| fold | Local CV | file |
| :--: | :--:     | :--  |
| 0    | 98.255402% | /mnt/hdd1/alcon2019/exp7/2019-07-13_12-25-44/fold0/ |
| 1    | 98.342150% | /mnt/hdd1/alcon2019/exp7/2019-07-13_12-25-44/fold1/ |
| 2    | 98.308331% | /mnt/hdd1/alcon2019/exp7/2019-07-18_10-17-48/fold2/ |
| 3    | 98.041016% | /mnt/hdd1/alcon2019/exp7/2019-07-22_13-55-51/fold3/ |
| 4    | 98.206913% | /mnt/hdd1/alcon2019/exp7/2019-07-22_13-55-51/fold4/ |

- Public score: 94.7000%



### EXP-8

---

 #### Detail

- Model : DenseNet201(pre-train ImageNet) + BiGRU x 2
- Batch Size : 40 (about 2000 iter / epoch)
- Resolution : 6 (192 x 128  <--- 210 x 150)
- Mixed-Precision Training (optim level '01')
- SGDR(5epoch / cycle)
- 3~10cycle (8cycle) で SnapShot Ensemble
- Total epoch: 50

#### Result

| fold | Local CV(single best) | file                                                |
| :--: | :-------------------: | :-------------------------------------------------- |
|  0   | 97.868174%(98.73813%) | /mnt/hdd1/alcon2019/exp8/2019-07-31_01-30-12/fold0/ |
|  1   |                       | /mnt/hdd1/alcon2019/exp8/2019-08-01_04-55-40/fold1/ |
|  2   |                       | /mnt/hdd1/alcon2019/exp8/2019-08-01_04-55-40/fold2/ |
|  3   |                       | /mnt/hdd1/alcon2019/exp8/2019-08-01_04-55-40/fold3/ |
|  4   |                       | /mnt/hdd1/alcon2019/exp8/2019-08-01_04-55-40/fold4/ |

- Public score: 



### EXP-9

------

#### Detail

- Model : Inception-v4(pre-train ImageNet) + BiGRU x 2
- Batch Size : 180 (530~540 iter / epoch)
- Resolution : 6 (192 x 128  <--- 210 x 150)
- Mixed-Precision Training (optim level '01')
- SGDR(5epoch / cycle)
- 3~10cycle (8cycle) で SnapShot Ensemble
- Total epoch: 50

#### Result

| fold | Local CV(single best) | file                                                |
| :--: | :-------------------: | :-------------------------------------------------- |
|  0   |  97.072905%(98.701%)  | /mnt/hdd1/alcon2019/exp9/2019-08-01_01-41-16/fold0/ |
|  1   |  98.188031%(98.789%)  | /mnt/hdd1/alcon2019/exp9/2019-08-01_11-03-24/fold1/ |
|  2   |  98.083335%(98.830%)  | /mnt/hdd1/alcon2019/exp9/2019-08-01_23-42-41/fold2/ |
|  3   |                       |                                                     |
|  4   |                       |                                                     |

- Public score: 



### EXP-11

------

#### Detail

- Model : SEResNeXt-101(pre-train ImageNet) + BiGRU x 2
- Batch Size : 64 (1500 iter / epoch)
- Resolution : 6 (192 x 128  <--- 210 x 150)
- Mixed-Precision Training (optim level '01')
- SGDR(5epoch / cycle)
- 3~10cycle (8cycle) で SnapShot Ensemble
- Total epoch: 50
- 遅い

#### Result

| fold | Local CV | file |
| :--: | :------: | :--- |
|  0   |          |      |
|  1   |          |      |
|  2   |          |      |
|  3   |          |      |
|  4   |          |      |

- Public score: 



## メモ

- resnet18,batch 128で16m / epoch
- se_resnext101+LSTM(unidirect)でK80(VRAM 11GB) 2枚だと(336x224)でbatch 16/GPUでOOM。4~ hour/epoch
- 30epochあれば十分かもしれない
- SSE: 10epoch(SGDR) + 5epoch * 4shot = 30epoch
- HorizonFlipでも行けるかもしれない(反転しても同じものは存在しないため)
- Attention Branch Networkを試して見たい（Wide ResNet, SENet, ResNeXtあたり) => 精度悪化
- SSE有効（5epoch/cycleだと足りないかも.でも5epochでもいいかも（？）)
- SENet効かない
- 独自モデルも欲しいよね（）

### 最終的なパイプライン
1. Resolution 6で学習
2. Exp-7, 8, 9, 10(OctResNet-50+DenseNet-201+Inception-v4+SE-ResNeXt-101)でのアンサンブル
3. 時間があれば、Pseudo-Labeling



スケジュール

| 日付                    | やること                        |
| ----------------------- | ------------------------------- |
| 2019/08/01 ~ 2019/08/05 | Exp-8                           |
| 2019/08/06 ~ 2019/08/10 | Exp-9                           |
| 2019/08/11 ~ 2019/08/15 | Exp-10                          |
| 2019/08/16 ~ 2019/08/17 | Pseudo-Label 作成               |
| 2019/08/18 ~ 2019/08/23 | Exp-7(Pseudo-Labeling Training) |
| 2019/08/24 ~ 2019/08/30 | Exp-8(Pseudo-Labeling Training) |
| 2019/08/31              | 最終日                          |

memo

Pseudo-Labelingの方が訓練データが大きので学習に時間がかかる。

時間が足りないのでクラウド使う。



#### 学習時のテクニック
- Dropout
- mixup
- optimizer: adam or sgd
- SGDR 

## アイデア
- backbone encoder + LSTM（or GRU）
- margin augmentation
- Resolution Ensemble

## TODO
- PreActOctResNet
- ShakeDrop

## Model(original)
- 001: Encoder-Decoder ResNet(test model)
- 002: SE_ResNeXt101+LSTM
- 003: ResNet50+LSTM
- 004: ResNet50+Residual LSTM
- 005: ResNet50+Residual LSTM+MLP
- 006: ResNet50+GRU
- 007: ResNet50+LSTM+Attention Branch Network
- 008: ResNet50+GRUx2
- 009: ResNet50+GRUx3
- 010: OctConv ResNet50 + BiGRUx2
- 011: SEResNeXt101 + BiGRUx2
- 012: OctResNet152 + BiGRUx2
- 013: OctConv PreAct ResNet50(Miss Implementation) + BiGRUx2
- 014: DenseNet201 + BiGRUx2
- 015: Inception-v4 + BiGRUx2 

## Model(backbone use)
- ResNet-18,34,50,101,152(5~7 day/model)
- ResNeXt-50,101 (2week/model)
- SENet (3week/model)
- DenseNet-101,201
- WideResNet
- Inception-v4
- NasNet large

## Model(backbone memo)
- ResNet
- PreAct ResNet
- ResNeXt
- WideResNet
- NasNet large
- SENet(SE-ResNet, SE-ResNeXt)
- DenseNet
- FishNet
- DarkNet
- Inception-v2,3,4 (+Inception-ResNet)
- EfficientNet
- pnasnet
- xception
- PolyNet
- OctConv
- PyramidNet
- DenseNet-BC


## Technique
- Shake-Shake
- Shake-Drop
- Stochastic Depth
- pruning
- distillation
- Negative Sampling
- AutoAugmentation
- ArcFace
- CosFace
- Pseudo-Label
- Dropout(p=0.3, 0.4, 0.5)
- Mixup(beta=0.2, 0.5)
- RICAP(beta=0.3)
- ICAP(beta=0.5)


## Scheduler
- Cosine Annearing
- WarmUp
- Step


## Optimizer
- momentum SGD
- Nestrov momentum SGD
- Adam
- RMSProp
- Adabound


## Loss Function
- Cross Entropy Loss
- Binary Cross Entropy Loss
- Focal Loss
- Focal Travesky Loss
- Lovasz Loss


## Ensemble
- Stacking(mean+vote, NN, GNN, CNN)
- SnapShot Ensemble
- Fast Geometric Ensembling
- Stochastic Weight Averaging

## Activation
- ReLU
- ELU
- GELU
- Swish
- Erase ReLU

#### Kind of mean
- 算術平均
- 加重平均
- 調和平均
- 幾何平均

## Image Resolution
1. 336 x 224  <--- 384 x 256
2. 168 x 112  <--- 192 x 128
3. 128 x 128  <--- 150 x 150
4. 84  x 56   <--- 96  x 64
5. 64  x 64   <--- 72  x 72
6. 192 x 128  <--- 210 x 150
