"""波数 kh ごとに G = cos^2(kh/2)（接線方向の面→セル→面の平均の利得の見積もり）、z = z_g (kh/pi)^3 とした走査。
G はあくまで見積もり（2 次元の射影と界面の帯の厚さは入れていない）。"""
import numpy as np, sys
sys.path.insert(0, "audit/d179")
from collocated_gain import radius
kh = np.linspace(0.02, np.pi, 400)
for zg in (0.5, 1.0, 2.0, 4.9):
    for G_on in (False, True):
        out = []
        for name in ("a:E1+avg", "c", "d1", "b1"):
            lam = [np.log(radius(name, zg * (k / np.pi) ** 3, np.cos(k / 2) ** 2 if G_on else 1.0)) for k in kh]
            i = int(np.argmax(lam))
            out.append("%s max %+.4f (kh/pi=%.2f)" % (name, lam[i], kh[i] / np.pi))
        print("z_g=%-4g %s: %s" % (zg, "G=cos^2" if G_on else "G=1    ", " | ".join(out)))
