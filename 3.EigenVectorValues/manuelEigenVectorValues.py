def determinant(matris):
    n = len(matris)
    if n == 1:
        return matris[0][0]
    if n == 2:
        return matris[0][0] * matris[1][1] - matris[0][1] * matris[1][0]
    det = 0
    for j in range(n):
        altMatris = []
        for i in range(1, n):
            satir = []
            for k in range(n):
                if k != j:
                    satir.append(matris[i][k])
            altMatris.append(satir)
        kofaktor = (-1) ** j * matris[0][j] * determinant(altMatris)
        det += kofaktor
    return det


def matrisCarpma(matris, skalar):
    return [[skalar * eleman for eleman in satir] for satir in matris]


def matrisCikarma(matris1, matris2):
    return [[matris1[i][j] - matris2[i][j] for j in range(len(matris1[0]))] for i in range(len(matris1))]


def birimMatris(n):
    matris = [[0] * n for _ in range(n)]
    for i in range(n):
        matris[i][i] = 1
    return matris


def karakteristik(matris, lambd):
    I = birimMatris(len(matris))
    I_lambd = matrisCarpma(I, lambd)
    return determinant(matrisCikarma(matris, I_lambd))


def ozdegerHesapla(matris):
    lambd_degerler = [-15 + i * 0.1 for i in range(1000)]  
    detDegerler = [karakteristik(matris, lambd) for lambd in lambd_degerler]

    detDegerler_mutlak = [abs(deger) for deger in detDegerler]

    min_index = detDegerler_mutlak.index(min(detDegerler_mutlak))
    return lambd_degerler[min_index]


def ozvektorHesapla(matris, lambd):
    I = birimMatris(len(matris))
    yeniMatris = matrisCikarma(matris, matrisCarpma(I, lambd))

    def satirIndirgeme(matris):
        satirSayisi = len(matris)
        sutunSayisi = len(matris[0])
        rank = 0

        for i in range(satirSayisi):
            if matris[i][i] == 0:
                for j in range(i + 1, satirSayisi):
                    if matris[j][i] != 0:
                        matris[i], matris[j] = matris[j], matris[i]
                        break

            if matris[i][i] != 0:
                for j in range(i + 1, satirSayisi):
                    if matris[j][i] != 0:
                        faktor = matris[j][i] / matris[i][i]
                        for k in range(i, sutunSayisi):
                            matris[j][k] -= faktor * matris[i][k]
                rank += 1

        return matris, rank

    matris, rank = satirIndirgeme(yeniMatris)
    if rank < len(matris):
        return [row for row in matris if row != [0] * len(matris[0])][0] 
    else:
        return None


matris = [
    [4, 1, 2, 3],
    [1, 2, 0, 1],
    [2, 0, 3, 1],
    [3, 1, 1, 4]
]

ozdeger = ozdegerHesapla(matris)
ozvektor = ozvektorHesapla(matris, ozdeger)

print("Özdeğer: ", ozdeger)
print("Özvektör: ", ozvektor)