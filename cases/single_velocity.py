"""One velocity field with a consistent discrete projection (docs/notes/48): the linear capillary-wave system as
matrices, for a cell-centred velocity with the exact projection and for a MAC (face) velocity.

Linearised about the flat interface at rest (density ratio 1, walls, exact linear curvature, band-corrected
kinematics). Free faces exclude the wall faces (zero normal velocity).

* cell:  u_c the unknown, u_f = I u_c (I the fd2 face average), D_c = D_f I, G_c = -D_c^T (equal volumes),
  Pi_c = 1 - G_c (D_c G_c)^+ D_c; viscosity L_c of the cell solver (fd2 or CCD full stress), force I^T a.
* MAC:   u_f the unknown, Pi_f = 1 - G_f (D_f G_f)^+ D_f with G_f = -D_f^T, viscosity the fd2 Laplacian of each face
  component (free slip on the tangential walls), force a on the faces.

Semi-discrete system ``u' = Pi (nu L u + f(psi))``, ``psi' = -T u_f``; its eigenvalue closest to the exact ``s``
gives the damping and frequency errors without any time-stepping error.
"""

from __future__ import annotations

import numpy as np

from cases.adjoint_kinematics import free_faces, from_face_field, to_face_field, transport_matrix


def setup(n=32, f=1.0, viscous="fd2"):
    from basilisk_cupy.terms.viscous import CCDViscosity, FD2Viscosity
    from cases.capwave_jacobian import CapwaveMap

    cm = CapwaveMap(n, f, curvature="exact", cls_kinematics="extension",
                    viscous=CCDViscosity() if viscous == "ccd" else FD2Viscosity())
    return cm


def operators(cm, viscous="fd2", eps=1e-6):
    from basilisk_cupy.terms.viscous import CCDStress, FD2Stress
    from cases.case_c import Stepper

    ns, tp, cls, grid = cm.ns, cm.tp, cm.cls, cm.grid
    n = grid.n[0]
    nc = n * n
    faces = free_faces(grid)
    nf = len(faces)
    h = float(grid.h[0])
    # interpolation (free faces x 2 nc), divergence (nc x nf)
    I = np.zeros((nf, 2 * nc))
    D = np.zeros((nc, nf))
    for k, (a, i, j) in enumerate(faces):
        if a == 0:
            I[k, (i - 1) * n + j] = 0.5
            I[k, nc + 0 + (i * n + j)] = 0.0
            I[k, i * n + j] = 0.5
            D[(i - 1) * n + j, k] += 1.0 / h
            D[i * n + j, k] -= 1.0 / h
        else:
            I[k, nc + i * n + (j - 1)] = 0.5
            I[k, nc + i * n + j] = 0.5
            D[i * n + (j - 1), k] += 1.0 / h
            D[i * n + j, k] -= 1.0 / h
    # careful: x-face (i, j) lies between cells (i-1, j) and (i, j); u_x components occupy the first nc entries
    I[:, :] = 0.0
    for k, (a, i, j) in enumerate(faces):
        off = 0 if a == 0 else nc
        if a == 0:
            I[k, off + (i - 1) * n + j] = 0.5
            I[k, off + i * n + j] = 0.5
        else:
            I[k, off + i * n + (j - 1)] = 0.5
            I[k, off + i * n + j] = 0.5
    # force: GFM face acceleration, linear in the psi perturbation (flat interface: a(psi0) = 0)
    st = Stepper(ns, tp, "be")
    psi0 = np.asarray(cls.psi.interior).copy()
    A = np.zeros((nf, nc))
    base = from_face_field(st.face_force(psi0), faces)
    for c in range(nc):
        d = np.zeros(nc)
        d[c] = eps
        A[:, c] = (from_face_field(st.face_force(psi0 + d.reshape(n, n)), faces) - base) / eps
    cls.psi.interior = psi0
    cls.psi.fill_ghosts()
    T, faces_t = transport_matrix(cls, "extension")
    assert faces_t == faces
    # cell viscosity (full stress of the cell solver, unit viscosity)
    op = CCDStress() if viscous == "ccd" else FD2Stress()
    tmp = ns.u.copy("Lc")
    Lc = np.zeros((2 * nc, 2 * nc))
    for c in range(2 * nc):
        e = np.zeros(2 * nc)
        e[c] = 1.0
        tmp[0].interior = e[:nc].reshape(n, n)
        tmp[1].interior = e[nc:].reshape(n, n)
        tmp.fill_ghosts()
        out = op(tmp, 1.0)
        Lc[:, c] = np.concatenate([np.asarray(out[0]).ravel(), np.asarray(out[1]).ravel()])
    # MAC viscosity: fd2 Laplacian of each face component; walls: normal component zero at its wall faces,
    # tangential free slip (mirror) at the other walls
    Lf = np.zeros((nf, nf))
    index = {fc: k for k, fc in enumerate(faces)}
    for k, (a, i, j) in enumerate(faces):
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + da, j + db
            nb = (a, ii, jj)
            along = 0 if da else 1  # axis of the neighbour step
            if nb in index:
                Lf[k, index[nb]] += 1.0 / h ** 2
                Lf[k, k] -= 1.0 / h ** 2
            else:
                if along == a:  # neighbour is a wall face of the same component: value 0 (Dirichlet at the face)
                    Lf[k, k] -= 1.0 / h ** 2
                else:  # beyond a tangential wall: mirror (Neumann), no contribution
                    pass
    return {"I": I, "D": D, "A": A, "T": T, "Lc": Lc, "Lf": Lf, "faces": faces, "nc": nc, "nf": nf}


def projection(D, G):
    S = D @ G
    return np.eye(G.shape[0]) - G @ np.linalg.pinv(S, rcond=1e-10) @ D, S


def semi_discrete(ops, nu, scheme="cell"):
    I, D, A, T, Lc, Lf = (ops[k] for k in ("I", "D", "A", "T", "Lc", "Lf"))
    nc = ops["nc"]
    if scheme == "cell":
        Dc = D @ I
        Gc = -Dc.T
        P, S = projection(Dc, Gc)
        Juu = P @ (nu * Lc) @ P
        Jup = P @ I.T @ A
        Jpu = -T @ I @ P
    else:
        Gf = -D.T
        P, S = projection(D, Gf)
        Juu = P @ (nu * Lf) @ P
        Jup = P @ A
        Jpu = -T @ P
    m = Juu.shape[0]
    J = np.block([[Juu, Jup], [Jpu, np.zeros((nc, nc))]])
    return J, P, S


def eigen(J, s):
    lam = np.linalg.eigvals(J)
    k = int(np.argmin(np.abs(lam - s)))
    return lam[k]


def step_eigen(J, s, dt, method):
    """Eigenvalue of one CN or SDIRK2 step of the linear system ``x' = J x`` closest to ``exp(s dt)``, as a rate."""
    import cmath

    n = J.shape[0]
    E = np.eye(n)
    if method == "cn":
        M = np.linalg.solve(E - 0.5 * dt * J, E + 0.5 * dt * J)
    elif method == "be":
        M = np.linalg.inv(E - dt * J)
    else:
        g = 1.0 - 1.0 / np.sqrt(2.0)
        R1 = np.linalg.inv(E - g * dt * J)
        K1 = J @ R1  # stage derivative from x
        M = np.linalg.solve(E - g * dt * J, E + (1 - g) * dt * K1)
    lam = np.linalg.eigvals(M)
    tg = cmath.exp(s * dt)
    k = int(np.argmin(np.abs(lam - tg)))
    return complex(np.log(lam[k]) / dt)


def interp_div(grid):
    """``I`` (free faces x 2 nc) and ``D`` (nc x free faces) without the force and the viscosity."""
    n = grid.n[0]
    nc = n * n
    faces = free_faces(grid)
    h = float(grid.h[0])
    I = np.zeros((len(faces), 2 * nc))
    D = np.zeros((nc, len(faces)))
    for k, (a, i, j) in enumerate(faces):
        off = 0 if a == 0 else nc
        if a == 0:
            I[k, off + (i - 1) * n + j] = I[k, off + i * n + j] = 0.5
            D[(i - 1) * n + j, k] += 1.0 / h
            D[i * n + j, k] -= 1.0 / h
        else:
            I[k, off + i * n + (j - 1)] = I[k, off + i * n + j] = 0.5
            D[i * n + (j - 1), k] += 1.0 / h
            D[i * n + j, k] -= 1.0 / h
    return I, D, faces
