import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.graph_objects as go 
import plotly.express as px 
import plotly.io as pio
from plotly.subplots import make_subplots


__all__ = ['Game']


def power_step(t, a=2, b=2, alpha=1, *args, **kvargs):
    return a / (b + t)**alpha


def const_step(t, c=0.005, *args, **kvargs):
    return c


class Game:   # God Object
    def __init__(self, payoff=None, p=None, q=None, pt=None, qt=None):
        self.payoff = payoff
        self.p = p
        self.q = q
        self.pt = pt
        self.qt = qt
        
    def read(self, file, **options):
        payoff = pd.read_csv(file, index_col=0, **options)
        self.__init__(payoff)
        return self
    
    def plot_payoff(self, origin='lower', mid=0, palette=px.colors.diverging.Armyrose_r, **options):
        fig = px.imshow(self.payoff, origin=origin, color_continuous_midpoint=mid, color_continuous_scale=palette, **options)
        fig.update_layout(
            xaxis = dict(
                tickmode = 'array',
                tickvals = np.arange(len(self.payoff.index)),
                ticktext = self.payoff.index
            ),
            yaxis = dict(
                tickmode = 'array',
                tickvals = np.arange(len(self.payoff.index)),
                ticktext = self.payoff.index
            ),
        )
        return fig
        
    def plot_structure(self, **options):
        df = self.payoff
        pca = PCA(n_components=2)
        xyz = pca.fit_transform(self.payoff)
        fig = px.scatter(x=xyz[:, 0], y=xyz[:, 1], text=df.index, **options)
        fig.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1))
        return fig      
    
    def plot_structure_3d(self, **options):
        df = self.payoff
        pca = PCA(n_components=3)
        xyz = pca.fit_transform(self.payoff)
        fig = px.scatter_3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], text=df.index, **options)
        return fig      
    
    def play(self, maxiter=100000, step_size=power_step, s1=0, s2=0, track=True, debug=False, **options):
        A = self.payoff
        m, n = A.shape
        basis1, basis2 = np.eye(m), np.eye(n)
        p = np.array(basis1[s1], dtype='float64')   # caution: instantialize new arrays, 
        q = np.array(basis2[s2], dtype='float64')   # caution: in-place operations below.

        if debug:
            print("p0={}    q0={} \n".format(p, q))
        if track:
            pt = np.zeros((maxiter+1, m))
            qt = np.zeros((maxiter+1, n))
            pt[0] = p
            qt[0] = q
        else:
            pt, qt = None, None

        for t in range(maxiter):
            if debug:
                print("t: {}".format(t))
            alpha = step_size(t, **options)
            dp = np.dot(A, q)
            i = np.argmax(dp)
            p += alpha * (basis1[i] - p)
            if debug:
                print("alpha={:.2f}    dp={}    i={}    p={}".format(
                    round(alpha, 2), dp.round(2), i, p.round(2)))

            dq = np.dot(p, A)
            j = np.argmin(dq)
            q += alpha * (basis2[j] - q)
            if debug:
                print("alpha={:.2f}    dq={}    j={}    q={}".format(
                    round(alpha, 2), dq.round(2), j, q.round(2)))

            if track:
                pt[t+1] = p
                qt[t+1] = q
        
        self.p, self.q, self.pt, self.qt = p, q, pt, qt
        self.eq = (p + q) / 2
        return self

    def plot_equilibrium(self, **options):
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
        fig.add_bar(x=self.payoff.index, y=self.p, name='P1', **options, row=1, col=1)
        fig.add_bar(x=self.payoff.columns, y=self.q, name='P2', **options, row=1, col=2)
        fig.update_yaxes(range=(0, 1))
        return fig
        
    def plot_play_ternary(self, **options):
        if len(self.p) != 3 or len(self.q) != 3:
            raise Exception("strategies should be limited to 3.")
        
        fig = go.Figure()
        fig.add_scatterternary(a=self.pt[:,0], b=self.pt[:,1], c=self.pt[:,2], name='P1', **options)
        fig.add_scatterternary(a=self.qt[:,0], b=self.qt[:,1], c=self.qt[:,2], name='P2', **options)
        label = self.payoff.index
        fig.update_layout(ternary={'aaxis_title': label[0], 'baxis_title': label[1], 'caxis_title': label[2]})
        return fig
        
    def plot_play_3d(self, marker_size=1, **options):
        pca = PCA(n_components=3)
        pca.fit(np.vstack((self.pt, self.qt)))
        pt = pca.transform(self.pt)
        qt = pca.transform(self.qt)
        fig = go.Figure()
        fig.add_scatter3d(x=pt[:, 0], y=pt[:, 1], z=pt[:, 2], name='P1', marker_size=marker_size, **options)
        fig.add_scatter3d(x=qt[:, 0], y=qt[:, 1], z=qt[:, 2], name='P2', marker_size=marker_size, **options)
        return fig
    
    def meta(self, epsilon=0.01, **options):
        dtype = [('policy', np.object_), ('proba', float)]
        meta = [(x, p) for x, p in zip(self.payoff.index, self.eq) if p>epsilon]
        meta = np.array(meta, dtype=dtype)
        meta.sort(order='proba')
        return meta[::-1]
 
    def plot_meta(self, **options):
        meta = list(zip(*self.meta(**options)))
        fig = go.Figure()
        fig.add_bar(x=meta[0], y=meta[1], **options)
        fig.update_yaxes(range=(0, 1))
        return fig
