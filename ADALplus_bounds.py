import numpy as np
import scipy.linalg as la
import time
from scipy.io import loadmat
from sksparse.cholmod import cholesky
import gurobi as gb
np.seterr(divide='ignore')


def dual_bound_init(AA, AAt, bb, CC, mleq, LL):
    A = AA.copy()
    At = AAt.copy()
    b = bb.copy()
    C = CC.copy()
    L = LL.copy()
    L_vec = L.reshape((-1, 1))
    sense = np.array(['>' if i != -np.inf else '=' for i in L_vec])
    L[L == -np.inf] = 0
    L_vec = L.reshape((-1, 1))

    Agurobi = At
    Agurobi[:, mleq:] = -Agurobi[:, mleq:]
    idx = Agurobi.getnnz(1) > 0
    Agurobi = Agurobi[Agurobi.getnnz(1) > 0]
    obj = np.vstack([A[0:mleq, :] * L_vec - b[0:mleq], b[mleq:] - A[mleq:, :] * L_vec])

    # Create Gurobi LP model
    model = gb.Model()
    model.setParam('outputflag', 0)
    model.ModelSense = gb.GRB.MAXIMIZE
    model.setParam('FeasibilityTol', 1e-5)
    y = model.addMVar(Agurobi.shape[1],
                      lb=[0.0 for i in range(mleq)] + [-float('inf') for i in range(Agurobi.shape[1] - mleq)],
                      obj=obj[:, 0], vtype=gb.GRB.CONTINUOUS, name="y")
    mc = model.addMConstr(Agurobi, y, sense[idx], (- C).reshape((-1, 1))[idx], name='dual')
    model.update()
    model.write('test.lp')
    res = {'model': model, 'C': C, 'L': L, 'idx': idx, 'mc': mc}
    return res


def dual_bound(model, Z):
    # lam, ev = la.eigh(Z)
    # Z = ev@ np.diag(lam) @ ev.T

    mc = model['mc']
    mc.setAttr("RHS", (Z - C).reshape((-1, 1))[model['idx']][:, 0])
    model['model'].update()
    model['model'].write('test.lp')
    model['model'].optimize()

    if model['model'].Status == gb.GRB.OPTIMAL:
        # a safe dual bound have been found
        p = model['C'] - Z
        lb = model['model'].ObjVal + np.sum(model['L'].reshape((1, -1)) @ p.reshape((-1, 1)))
    else:
        lb = -np.inf

    return lb


def error_bound(dual, Aty, C, X, S, y, mleq, mu=1.1, max_lamb_x=None):
    n = C.shape[0]
    # If no upper bound on the maximum eigenvalue of the optimal X is given,
    # then it is estimated by the current X, scaled by a factor mu
    max_lamb_x = mu * np.max(la.eigh(X)[0]) if not max_lamb_x else max_lamb_x

    # Compute a feasible Znew (which in general will not be Positive Semidefinite
    Znew = C - Aty - S
    znew = -y[:mleq]
    lb0 = dual
    pert = 0.
    lam, _ = la.eigh(Znew)
    I = np.where(lam < 0)[0]

    if I.any():
        pert += max_lamb_x * np.sum(lam[I])

    I = np.where(znew < 0)[0]
    if I.any():
        pert += max_lamb_x * np.sum(znew[I])

    lb = lb0 + pert
    return lb, max_lamb_x


def norm_bound(dual, K, U, debug=False):
    bound = dual - U * K
    if debug:
        print('Norm Bound: %d' % U)
        print('Safe dual bound: %13.4f' % bound)
    return bound


def project(X, L, U):
    # Works only with Numpy Data types
    if X.shape == ():
        return L if X < L else (U if X > U else X)
    idx_l = X < L
    idx_u = X > U
    X[idx_l] = L[idx_l]
    X[idx_u] = U[idx_u]
    return X

def ADMM_bounds(A, b, C, mleq, L, sigma=1., options={}):
    # Initialization
    tstart = time.time()
    # Read option from dict
    tol = 1e-5 if 'tolerance' not in options else options['tolerance']
    max_iter = 1000 if 'max_iter' not in options else options['max_iter']
    timelimit = 3600 if 'timelimit' not in options else options['timelimit']
    print_it = 100 if 'print_it' not in options else options['print_it']
    debug = True if 'debug' not in options else options['debug']
    num_iter = 1
    done = False
    result = {}

    # initialization of sigma box
    t_min = np.float64(1e-4)
    t_max = np.float64(1e+7)

    m, n2 = A.shape
    n = int(np.sqrt(n2))

    assert L is None or L.shape[0] == n, 'mismatch dimension on bounds L'

    At, AAT = A.T, A @ A.T

    AAT_lil = AAT.tolil()
    # Add slack variables to mleq constraints
    AAT_lil.setdiag(AAT_lil.diagonal() + np.array([1 if i < mleq else 0 for i in range(m)]))
    AAT = AAT_lil.tocsc()

    factor = cholesky(AAT)
    secs = time.time() - tstart

    if debug:
        print('Cholesky factorization completed after: %12.5f' % secs)

    # initialize primal variable
    Y = np.zeros((n, n))
    # initialize dual variable
    Z = np.zeros((n, n))
    # initialize S, dimension of S: (n) by (n) ... multipliers of X>=L
    S = np.zeros((n, n))

    # Slack variables of the primal, dimension of x: (mleq) by (1)
    # we pad x to dimension (m) by (1)
    x = np.zeros((m, 1))
    # Surplus variables of the primal, dimension of z: (mleq) by (1)
    # we pad z to dimension (m) by (1)
    z = np.zeros((m, 1))

    idx = L!=-np.inf

    normb, normC = la.norm(b), la.norm(C)

    # Needed to initialize the LP for dual bound safe bound
    # model = dual_bound_init(A, At, b, C, mleq, L)

    if debug:
        print(' it     secs       dual        primal       dFeas    pFeas     X>=L     compXS   sigma  ')

    while not done:
        # weight for sigma update
        w = np.power(2, -(num_iter - 1) / 100)

        # given Y, Z, S and sigma, solve for y
        M_tmp = Y/sigma - C + Z + S
        rhs = b/sigma - A*M_tmp.reshape((-1, 1)) - x/sigma - z
        y = factor(rhs)
        Aty = (At * y).reshape((n, n))

        S = C - Aty - Z - Y / sigma + L / sigma
        S[S < 0] = 0

        M = Aty - C + Y / sigma + S

        M1 = x / sigma + y
        M1 = M1[:mleq]

        lam, ev = la.eigh(M)
        I = np.where(lam > 0)[0]
        j = len(I)
        if j < n / 2:
            evp = np.zeros((n, j))
            for r in range(j):
                ic = I[r]
                evp[:, r] = ev[:, ic] * np.sqrt(lam[ic])
            if j == 0:
                evp = np.zeros((n, 1))
            Mp = evp @ evp.T
            Mn = M - Mp
        else:
            I = np.where(lam < 0)[0]
            j = len(I)  # should be <= n/2
            evn = np.zeros((n, j))
            for r in range(j):
                ic = I[r]
                evn[:, r] = ev[:, ic] * np.sqrt(-lam[ic])
            if j == 0:
                evn = np.zeros((n, 1))
            Mn = -evn @ (evn.T)
            Mp = M - Mn

        # Project the diagonal part of M
        mp = M1.copy()
        mp[mp < 0] = 0
        mn = M1 - mp

        Z = -Mn
        z[:mleq] = -mn
        X = sigma * Mp
        x[:mleq] = sigma * mp

        # Y update
        Y = X

        g = b - A * Y.reshape((-1, 1)) - x
        G = C - Aty - Z - S
        gg = - y - z
        gg = gg[:mleq]

        normX = la.norm(X)

        err_d = (la.norm(G) + la.norm(gg))/(1 + normC)
        dual = ((b.T @ y) + np.sum(L[idx].reshape((1, -1)) @ S[idx].reshape((-1, 1)))).item()
        err_p = la.norm(g)/(1 + normb)
        primal = np.sum(C.reshape((1, -1)) @ Y.reshape((-1, 1)))
        rel_err_p, rel_err_d = err_p / (1 + normb), err_d / (1 + normC)
        err_X_L = la.norm(X - np.maximum(L, X))/normX
        XL = X[idx] - L[idx]
        compXS = np.abs(np.sum(S[idx].reshape((1, -1)) @ XL.reshape((-1, 1))))/ (1+la.norm(XL) + la.norm(S))
        secs = time.time() - tstart
        num_iter = num_iter + 1

        # Printing
        if (num_iter % print_it) == 0:
            if debug:
                print('%3.0d %8.2f %13.5e %13.5e %8.3f %8.3f  %8.3f %8.3f  %9.6f' %
                      (num_iter, secs, dual, primal, np.log10(rel_err_d), np.log10(rel_err_p), np.log10(err_X_L),
                       np.log10(compXS), sigma))

        # Stopping criteria
        if np.max([rel_err_d, rel_err_p, err_X_L, compXS]) < tol or num_iter > max_iter or secs > timelimit:
            if debug:
                print('%3.0d %8.2f %13.5e %13.5e %8.3f %8.3f  %8.3f %8.3f  %9.6f' %
                      (num_iter, secs, dual, primal, np.log10(rel_err_d), np.log10(rel_err_p), np.log10(err_X_L),
                       np.log10(compXS), sigma))

            if debug:
                print('total time: %10.3f' % secs)
            if num_iter > max_iter:
                if debug:
                    print('max outer iterations reached.')
            if secs > timelimit:
                if debug:
                    print('Time limit exceeded')

            done = 1

        ratio = (la.norm(X) + la.norm(x)) / (la.norm(Z) + la.norm(z))
        sigma = (1 - w) * sigma + w * project(ratio, t_min, t_max)
    # Calls to safe bounding procedures
    # db = dual_bound(model, Z)
    # eb = error_bound(dual, Aty, C, X, S, y, mleq)
    # nb = norm_bound(dual, (la.norm(G) + la.norm(gg)), U)
    return result

# Example usage
h = loadmat(os.path.join('instances', 'test_adal.mat'))
A, b, C, mleq, L = h['A'], h['b'], h['C'], h['mleq'].sum(), h['L']
ADMM_bounds(A, b, C, mleq, L, sigma=50., options={'tolerance' : 1e-6})