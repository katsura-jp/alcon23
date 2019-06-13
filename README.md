# 第23回　アルゴリズムコンテスト
テーマ：三文字の崩し文字認識

## 手順
1. データ詳細テーブルと変換器を生成する。
```
$ cd src
$ python creat_vocab.py # make vocabrary
$ python creat_table_multiprocess.py # make meta files
$ cd ../
```
2. 学習をする。
```
$ cd experiment
$ vim ../param/expN.yaml # change params
$ python expN.py
```

3. 推論する。




## 結果
| exp No. | Local CV | Public | Private | comment |
| ------: | -------: | -----: | ------: | :------ |
| 0       |          |        |         | example |
| 1       |          |        |         |         |
| 2       |          |        |         |         |
| 3       |          |        |         |         |
| 4       |          |        |         |         |
| 5       |          |        |         |         |
| 6       |          |        |         |         |
| 7       |          |        |         |         |
| 8       |          |        |         |         |


## アイデア
- backbone encoder + LSTM


## TODO
- Predictionのスクリプト作成
- logの追加
- Dropout(p=0.3, 0.4, 0.5)の検証
- Mixup(beta=0.2, 0.5)の検証
- RICAP(beta=0.3)の検証
- ICAP(beta=0.5)の検証


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


## Technique
- Shake-Shake
- Shake-Drop
- Stochastic Depth
- Dropout
- pruning
- distillation
- Negative Sampling
- AutoAugmentation
- ArcFace
- CosFace

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


