"""STEP 3 - two-minute installation check."""
import numpy as np
from n1 import load, ENC, decode, train, rmse
Xtr, ytr, Xva, yva, Xte, yte = load("FD001")
print("FD001 windows: train", Xtr.shape, "validation", Xva.shape, "test", Xte.shape)
p, predict, best, n = train("lstm", Xtr, ytr, Xva, yva, seed=0, epochs=3)
print("LSTM after 3 epochs: test RMSE =", round(rmse(np.clip(predict(p, Xte), 0, 125), yte), 2), "(anything around 17-26 is fine)")
print("OK - installation works. Next: python step4_fig1_encodings.py")
