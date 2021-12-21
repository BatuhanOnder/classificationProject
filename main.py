import matplotlib.pyplot as plt
import numpy as np
import pandas

df = pandas.read_csv('data_456.csv', header=None)
print(df)

cikis = df.iloc[0:125, 5].values
cikis = np.where(cikis == 1, -1, 1)
giris = df.iloc[0:125, [1, 3]].values

plt.title('2D görünüm', fontsize=16)

plt.scatter(giris[:63, 0], giris[:63, -1], color='black', marker='o', label='birler')
plt.scatter(giris[63:125, 0], giris[63:125, 1], color='green', marker='x', label='sıfırlar')
plt.xlabel('sapel length')
plt.ylabel('petal length')
plt.legend(loc='upper left')

plt.show()


class Perceptron(object):
    def __init__(self, ogrenme_orani=0.1, iter_sayisi=10):
        self.ogrenme_orani = ogrenme_orani
        self.iter_sayisi = iter_sayisi

    def ogren(self, X, y):
        self.w = np.zeros(1 + X.shape[1])
        self.hatalar = []
        for _ in range(self.iter_sayisi):
            hata = 0
            for xi, hedef in zip(X, y):
                degisim = self.ogrenme_orani * (hedef - self.tahmin(xi))
                self.w[1:] += degisim * xi
                self.w[0] += degisim
                hata += int(degisim != 0.0)
            self.hatalar.append(hata)
        return self

    def net_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def tahmin(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)


siniflandirici = Perceptron(ogrenme_orani=0.1, iter_sayisi=10)

print(siniflandirici.ogren(giris, cikis))

print(siniflandirici.w)

print(siniflandirici.hatalar)

plt.plot(range(1, len(siniflandirici.hatalar) + 1), siniflandirici.hatalar)
plt.xlabel('Deneme')
plt.ylabel('Hatalı tahmin sayısı')
plt.show()