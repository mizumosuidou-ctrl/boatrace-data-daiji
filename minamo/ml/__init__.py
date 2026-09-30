"""LightGBMによるスタート順位ベースの予想。

raw CSV（deploy/ml_refresh.sh でデータベースから書き出したもの）から
  dataset.build   特徴量表（レースより前の情報だけで作る）
  train.run       学習・検証・保存
  live.MLPredictor 当日の出走表・直前情報から1着確率を出す
"""
