# get the best score as a string
def getMetricPostfix(m, h):
  vloss = h.history['val_loss'][0]
  vmetric = h.history[m][0]

  for l, d in zip(h.history['val_loss'], h.history[m]):
    if l < vloss:
      vloss = l
      vmetric = d

  return str(int(vmetric*1000))
