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
2. 学習/推論をする。
```
$ cd experiment
$ vim ../param/expN.yaml # change params
$ python expN.py
```

3. アンサンブル




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
- OctConv
- PyramidNet
- DenseNet-BC

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
- Pseudo-Label

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


