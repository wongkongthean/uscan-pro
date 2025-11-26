import numpy as np, pandas as pd, yfinance as yf, json
stocks = ["0700.HK", "9988.HK"]
data = yf.download(stocks, period="2y", progress=False)['Adj Close']
returns = np.log(1 + data.pct_change()).dropna()
mu = returns.mean().values * 252
sigma = returns.std().values * np.sqrt(252)
corr = returns.corr().values
L = np.linalg.cholesky(corr + np.eye(2)*1e-8)
T, N = 252, 5000
dt = 1/252
coupon, KI = 0.13, 0.70
S0 = data.iloc[-1].values
payoffs = np.full(N, 1 + coupon)
Z = np.random.normal(0,1,(2,T,N))
dW = np.tensordot(L, Z, axes=1) * np.sqrt(dt)
S = S0[:,None,None] * np.exp(np.cumsum((mu - 0.5*sigma**2)[:,None,None]*dt + sigma[:,None,None]*dW, axis=1))
worst = np.min(S / S0.min(), axis=0)
knocked = np.any(worst < KI, axis=0)
payoffs[knocked] = worst[-1, knocked] / S0.min()
er = (payoffs.mean() - 1) * 100
loss = (payoffs < 0.95).mean() * 100
json.dumps({"er": round(float(er),2), "loss": round(float(loss),2), "payoffs": payoffs[:800].tolist()})
