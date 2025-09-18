#!/usr/bin/env python3
# Imports nécessaires (numpy, matplotlib, etc.)
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
# Fonctions utiles (si besoin)

def calculer_temperature(R , A , B, C):
    inv_T = A + B*np.log(R) + C*(np.log(R))**3
    T = 1/(inv_T)
    return T - 273

def calculer_resistance(T_K , A , B , C):
    x = 1/C * (A - 1/T_K)
    y = np.sqrt((B/(3*C))**3 + (x/2)**2)

    R = np.exp( (y - x/2)**(1/3) - (y + x/2)**(1/3)  )
    return R


def linreg_fit(x, y):
    m, b = np.polyfit(x, y, 1)
    yhat = m*x + b
    rss = np.sum((y - yhat)**2)
    return m, b, rss

def piecewise_linreg(x, y, rmse_thresh=1e-2, min_len=5):
    """
    Greedy segmentation that guarantees continuous coverage of the
    whole x-array.  Each new segment starts where the previous ended.
    Returns a list of dicts:
      {'start':i0,'end':i1,'m':m,'b':b}
    with segments covering [0, len(x)).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)

    segs = []
    i0 = 0
    while i0 < n:
        # try to extend as far as we can while RMSE stays below threshold
        i1 = i0 + min_len
        last_good = None
        while i1 <= n:
            m, b = np.polyfit(x[i0:i1], y[i0:i1], 1)
            yhat = m*x[i0:i1] + b
            rmse = np.sqrt(np.mean((y[i0:i1] - yhat)**2))
            if rmse <= rmse_thresh:
                last_good = (m, b, i1)
                i1 += 1
            else:
                break
        if last_good is None:
            # if we cannot even fit min_len points, force a minimal segment
            last_good = np.polyfit(x[i0:i0+min_len], y[i0:i0+min_len], 1), i0+min_len
            m, b = last_good[0]
            j = last_good[1]
        else:
            m, b, j = last_good
        segs.append({'start': i0, 'end': j, 'm': m, 'b': b})
        i0 = j
        print(j)
        if i0 == len(x) - 1:
            break# next segment starts exactly where the previous ended
    return segs



def main(A , B, C):
    temperatures_c = np.array([i  for  i in range(0,105,5)])
    #print(len(temperatures_c))
    temperatures_k = np.array([i + 273 for i in temperatures_c])
    resistances = calculer_resistance(temperatures_k , A , B , C)
    temperatures_resultat_c = calculer_temperature(resistances , A , B , C)

    diff = [temperatures_c[i] - temperatures_resultat_c[i] for i in range(len(temperatures_c))]


    idx_t_pl = []
    segs = piecewise_linreg(temperatures_c , resistances)
    #print(segs)
    
    eps = 0.02
    for seg in segs:
        start = seg["start"]
        end = seg["end"]
        m = seg["m"]
        b = seg["b"]

        #if end > len(temperatures_c-1):
        #    end = len(temperatures_c) - 1


        slice = temperatures_c[start:end]
        line = b + m*slice



        idx = list(filter(lambda i : abs((resistances[i] - line[i-start])/resistances[i]) < eps, range(start , end )))
        idx_t_pl.extend(idx)


    print("Temperatures avec R Presque Lineare" , temperatures_c[idx_t_pl])

    for k, s in enumerate(segs, start=1):
        i0, i1 = s['start'], s['end']
        xs = temperatures_c[i0:i1]
        plt.plot(xs,
                s['m']*xs + s['b'],
                linewidth=2,
                linestyle = "dashed",
                color = "black")



    plt.plot(temperatures_c , resistances)
    plt.xlabel("Temperatures °C")
    plt.ylabel(r"Resistance Ohm $\Omega$")
    plt.title("Steinhart–Hart equation with piecewise linear regression")




    plt.legend()
    plt.show()

    pass


if __name__ == "__main__":

    A = 0.001129148
    B = 0.000234125
    C = 8.76741e-8



    main(A , B , C)