from dfbi import fingerprints as fp, metrics as mt, viz
t1 = "В этот же вечер князь Андрей Болконский..."
t2 = "Идёт по базару полицейский надзиратель Очумелов..."
M1 = fp.fingerprint(t1, horizon=3, decay=('exp',0.5), mask='letters')
M2 = fp.fingerprint(t2, horizon=3, decay=('exp',0.5), mask='letters')
print('L1 =', mt.dist_l1(M1, M2))
viz.heatmap(M1, 'Text1')
viz.heatmap(M2, 'Text2')
