# 第23回　アルゴリズムコンテスト
テーマ：三文字の崩し文字認識

## 概要
- 公式HP: https://sites.google.com/view/alcon2019/%E3%83%9B%E3%83%BC%E3%83%A0?authuser=0
- 提出場所: https://competitions.codalab.org/competitions/23101#participate-submit_results
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


## 結果
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
| 4.6 (2019-06-30_07-33-17) |           |       |        | ResNetGRU3 |  2 | Grad clip(1.0) |

## メモ
- resnet18,batch 128で16m / epoch
- se_resnext101+LSTM(unidirect)でK80(VRAM 11GB) 2枚だと(336x224)でbatch 16/GPUでOOM。4~ hour/epoch
- 30epochあれば十分かもしれない
- SSE: 10epoch(SGDR) + 5epoch * 4shot = 30epoch
- HorizonFlipでも行けるかもしれない(反転しても同じものは存在しないため)
- Attention Branch Networkを試して見たい（Wide ResNet, SENet, ResNeXtあたり) => 精度悪化
- 残り1サブのみなので注意

### 最終的なパイプライン
1. KANAデータで事前学習(resnet, se_resnextあたり) (6月中) (解像度とCutoutを考慮したモデルを作成)
2. low resolution(backbone+(Residual LSTM or GRU (bidirectional)))で学習,推論 (7月末まで)
3. Pseudo Labeling (7月末まで)
4. high resolution + SSE で学習、推論 (8/28まで)
5. Post Processing(Ensemble) (8/31まで)

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
- マイナーアップサンプリング
- validデータ結果を最終的に出力する。
- SSE用の実験コード作成
- SEResNetとResNetの比較

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
