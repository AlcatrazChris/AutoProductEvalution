import torch
import tensorwatch as tw
from train import LSTMModel

# 其实就两句话
model =LSTMModel(100)
tw.draw_model(model, [13859,100,128,3])