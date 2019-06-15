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
| exp No. | Local CV | Public | Private | comment |
| ------: | -------: | -----: | ------: | :------ |
| 0       |  87.391% |        |         | test example. only fold0. |
| 1       |          |        |         | simple resnet18.  |
| 2       |          |        |         |         |
| 3       |          |        |         |         |
| 4       |          |        |         |         |
| 5       |          |        |         |         |
| 6       |          |        |         |         |
| 7       |          |        |         |         |
| 8       |          |        |         |         |


## メモ
- resnet18,batch 128で16m / epoch
- se_resnext101+LSTM(unidirect)でK80(VRAM 11GB) 2枚だと(336x224)でbatch 16/GPUでOOM。4~ hour/epoch
- 30epochあれば十分かもしれない

### 最終的なパイプライン
1. KANAデータで事前学習(resnet, se_resnextあたり) (6月中)
2. low resolutionで学習,推論 (7月末まで)
3. Pseudo Labeling (7月末まで)
4. high resolution + SSE で学習、推論 (8/28まで)
5. Post Processing(Ensemble) (8/31まで)

#### 学習時のテクニック
- Dropout
- mixup
- optimizer: adam or sgd
- SGDR 

## アイデア
- backbone encoder + LSTM
- margin augmentation

## TODO
- Encoder-Decoder ResNet
- margin augment
- Mixup


## Model(backbone)
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
