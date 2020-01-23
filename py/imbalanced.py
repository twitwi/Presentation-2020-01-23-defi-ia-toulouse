
# %%

import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
import sklearn.neighbors


release = True
def mysavefig(fname, name=None, final=False, **kwargs):
    if release and not final: return
    if name is None: name = fname[:-4]
    if not final:
        txt = plt.figtext(0, 0, name, alpha=.3)
    plt.savefig('out/'+fname, bbox_inches='tight')
    if not final:
        txt.remove()


# %% PLOT THE LOG LOSS AND ITS GRADIENT

f = lambda x: -np.log2(x)
fp = lambda x: -1/x/np.log(2)

x = np.hstack([np.linspace(0.0001, .001, 10), np.linspace(0.001, .01, 10), np.linspace(0.01, .1, 20), np.linspace(.1,1,20)])
plt.figure(figsize=(12, 4))
plt.plot(x, f(x), label="- log($p_{y_i}$)")
plt.plot(x, fp(x), label=" $\\nabla$ -log($p_{y_i}$)")
x = np.array([.8, .2, .5])
plt.scatter(x, f(x))
plt.scatter(x, fp(x))
plt.grid()
plt.tight_layout()
plt.legend()
plt.ylim(-20, 10)
mysavefig('mle-grad.svg', final=True)
mysavefig('mle-grad.png')
plt.show()

# %%
print(fp(np.array([0.2, 0.5, 0.8])))
print(fp(1))





# %%

def bayes_risk(k=1, n_p=10, ratio=9, n_n=None, plot=True, show=True):
    if n_n is None: n_n = n_p * ratio
    Xp = np.random.uniform(0, 1, (n_p, 2))
    Xm = np.random.uniform(0, 1, (n_n, 2))
    X = np.vstack([Xp, Xm])
    Y = np.array([1]*Xp.shape[0] + [0]*Xm.shape[0])
    if plot:
        plt.scatter(Xm[:,0], Xm[:,1], color='blue', alpha=0.4, marker='_')
        plt.scatter(Xp[:,0], Xp[:,1], color='green', marker='+')

    nn = sklearn.neighbors.KNeighborsClassifier(k)
    nn = nn.fit(X, Y)

    l = lambda n: np.linspace(0, 1, n)
    xx, yy = np.meshgrid(l(100), l(100))
    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    if plot:
        plt.contour(xx, yy, Z, levels=[0.5])
        if show:
            plt.show()

    return Z.mean()

    #test = np.random.uniform(0, 1, (100000, 2))
    #dists, inds = nn.kneighbors(test)
    #print('Expected:', Y.mean(), 'empirical value:', Y[inds[:,0]].mean())

# %% LONG LONG

ks = [1,3,5]
n_ps = [10, 50, 100, 200, 1000]
runs = list(range(10))

res = np.zeros((len(ks), len(n_ps), len(runs)))
for ik,k in enumerate(ks):
    for in_p,n_p in enumerate(n_ps):
        for ir,r in enumerate(runs):
            res[ik,in_p,ir] = bayes_risk(k, n_p, plot=False)


# %%
#plt.imshow(res.mean(axis=2))
#plt.colorbar()
plt.plot(n_ps, res[0,:,:].mean(axis=1))
plt.plot(n_ps, res[1,:,:].mean(axis=1))
plt.plot(n_ps, res[2,:,:].mean(axis=1))

# %% generate positive and negative points (enough to subsample them later)
margin = .05/2 # half of it
XXp = np.random.uniform(0, 0.5, (10000, 2)) * [[1-margin*2, 2]]
XXm = np.random.uniform(0, 0.5, (10000, 2)) * [[1-margin*2, 2]] + [[0.5+margin, 0]]


# %% draw knn boundaries
def separated(k=1, n_p=10, ratio=9, n_n=None, plot=True, show=True, save=False, name=None, **kwargs):
    if n_n is None: n_n = n_p * ratio
    Xp = XXp[:n_p,:]
    Xm = XXm[:n_n,:]
    X = np.vstack([Xp, Xm])
    Y = np.array([1]*Xp.shape[0] + [0]*Xm.shape[0])

    nn = sklearn.neighbors.KNeighborsClassifier(k)
    nn = nn.fit(X, Y)

    #test = np.random.uniform(0, 1, (100000, 2))
    #dists, inds = nn.kneighbors(test)
    #print('Expected:', 0.5, 'empirical value:', Y[inds[:,0]].mean())

    l = lambda n: np.linspace(0, 1, n)
    xx, yy = np.meshgrid(l(100), l(100))
    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    if plot:
        plt.figure(figsize=(12, 12))
        plt.tight_layout()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        ctr = plt.contour(xx, yy, Z, levels=[0.5])
        ctrx = ctr.allsegs[0][0][:,0]
        ctry = ctr.allsegs[0][0][:,1]
        plt.fill_between([0.5-margin, 0.5+margin], [-1, -1], [1, 1], color=[.9,.9,.9,1])
        plt.fill_betweenx(ctry, ctrx, x2=0.5-margin, where=ctrx<0.5-margin, color=[1,.95,.95,1])
        plt.fill_betweenx(ctry, ctrx, x2=0.5+margin, where=ctrx>0.5+margin, color=[1,.95,.95,1])
        plt.scatter(Xp[:,0], Xp[:,1], marker='+')
        plt.scatter(Xm[:,0], Xm[:,1], alpha=0.4, marker='.')
        if save:
            if name is None:
                name = 'knn-boundary-%d-%d-%d'%(k, n_p, ratio)
            mysavefig(name+'.svg', **kwargs)
            mysavefig(name+'.png', **kwargs)
        if show:
            plt.show()

if not release:
    for k in [1,5]:
        for n_p in [10, 100, 1000]:
            for ratio in [10, 100]:
                separated(k, n_p, ratio, save=True)

    separated(11, 100, 10, save=True)
    separated(11, 20, 10, save=True)
    separated(11, 10, 10, save=True)

# %% RELEASE

if release:
    for k in [1]:
        for n_p in [10, 100, 1000]:
            for ratio in [10]:
                separated(k, n_p, ratio, save=True, final=True)
    for k in [11]:
        for n_p in [10, 100]:
            for ratio in [10]:
                separated(k, n_p, ratio, save=True, final=True)



# %% re-γNN
def go(N, filename):
    y = np.linspace(-0.87, 0.87, 1003)[None, :, None]
    x = np.linspace(-1.3, 1.3, 901)[:, None, None]

    xp = np.array([0, 0.66])[None,None,:]
    yp = np.array([0, 0])[None,None,:]
    rangle = np.random.uniform(0, np.pi*2, (1, 1, N))
    rradius = np.random.uniform(1, 1.7, rangle.shape)
    xm = rradius * np.cos(rangle)
    ym = rradius*4/6 * np.sin(rangle)
    d1m = np.min( ((x-xm)**2 + (y-ym)**2)**0.5, axis=2)
    d1p = np.min( ((x-xp)**2 + (y-yp)**2)**0.5, axis=2)
    s = d1m/(d1p+.0001)

    im = np.transpose(s)
    im2 = np.copy(im)

    CS = plt.contour(im2, extent=[x.min(), x.max(), y.min(), y.max()], levels=[1], colors=['k'], linewidths=[3])
    CS = plt.contour(im2, extent=[x.min(), x.max(), y.min(), y.max()], levels=list(np.arange(.2, 1, .2))+[2])
    plt.clabel(CS)
    plt.xlim([x.min(), x.max()])
    plt.ylim([y.min(), y.max()])
    plt.scatter(xp, yp, marker='+')
    plt.scatter(xm, ym, marker='.')
    plt.xticks([]) ; plt.yticks([])
    mysavefig(filename)
    plt.show()

go(300, 'gen-1-gamma-surrounded.pdf')
go(60, 'gen-1-gamma-surrounded-sparse.pdf')


# %% same as above but with gamma NN
def separated(k=1, n_p=10, ratio=9, n_n=None, plot=True, show=True, save=False, name=None, gammann=True, ne=ne, **kwargs):
    if k>1 and gammann:
        print("with γnn, unimplemented k>1")
    if n_n is None: n_n = n_p * ratio
    Xp = XXp[:n_p,:]
    Xm = XXm[:n_n,:]
    X = np.vstack([Xp, Xm])
    Y = np.array([1]*Xp.shape[0] + [0]*Xm.shape[0])

    nn = sklearn.neighbors.KNeighborsClassifier(k)
    nn = nn.fit(X, Y)

    #test = np.random.uniform(0, 1, (100000, 2))
    #dists, inds = nn.kneighbors(test)
    #print('Expected:', 0.5, 'empirical value:', Y[inds[:,0]].mean())

    l = lambda n: np.linspace(0, 1, n)
    xx, yy = np.meshgrid(l(100), l(100))
    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    if plot:
        plt.figure(figsize=(12, 12))
        plt.tight_layout()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        ctr = plt.contour(xx, yy, Z, levels=[0.5])
        plt.fill_between([0.5-margin, 0.5+margin], [-1, -1], [1, 1], color=[.9,.9,.9,1])
        if gammann:
            y = np.linspace(0, 1, 1003)[None, :, None]
            x = np.linspace(0, 1, 901)[:, None, None]
            xm = Xm[None,None,:,0]
            ym = Xm[None,None,:,1]
            xp = Xp[None,None,:,0]
            yp = Xp[None,None,:,1]
            gy = np.linspace(-0.87, 0.87, 1003)[None, :, None]
            gx = np.linspace(-1.3, 1.3, 901)[:, None, None]
            if ne is None:
                d1m = np.min( ((x-xm)**2 + (y-ym)**2), axis=2)**0.5
                d1p = np.min( ((x-xp)**2 + (y-yp)**2), axis=2)**0.5
                s = d1m/(d1p+.0001)
            else:
                a = (x-xm)**2
                b = (y-ym)**2
                d1m = ne.evaluate('min(a+b, 2)')**0.5
                a = (x-xp)**2
                b = (y-yp)**2
                d1p = ne.evaluate('min(a+b, 2)')**0.5
                #d1p = ne.evaluate('min(((x-xp)**2 + (y-yp)**2)**0.5, 2)')
                s = d1m/(d1p+.0001)


            im = np.transpose(s)
            im2 = np.copy(im)

            CS = plt.contour(im2, extent=[x.min(), x.max(), y.min(), y.max()], levels=[1], colors=['k'], linewidths=[3])
            CS = plt.contour(im2, extent=[x.min(), x.max(), y.min(), y.max()], levels=list(np.arange(.2, 1, .2))+[2])
            plt.clabel(CS)
            plt.xlim([x.min(), x.max()])
            plt.ylim([y.min(), y.max()])
            plt.scatter(xp, yp, marker='+')
            plt.scatter(xm, ym, marker='.')
            plt.xticks([]) ; plt.yticks([])
        else:
            ctrx = ctr.allsegs[0][0][:,0]
            ctry = ctr.allsegs[0][0][:,1]
            plt.fill_betweenx(ctry, ctrx, x2=0.5-margin, where=ctrx<0.5-margin, color=[1,.95,.95,1])
            plt.fill_betweenx(ctry, ctrx, x2=0.5+margin, where=ctrx>0.5+margin, color=[1,.95,.95,1])
            plt.scatter(xp, yp, marker='+')
            plt.scatter(xm, ym, marker='.')
        #plt.scatter(Xm[:,0], Xm[:,1], color='blue', alpha=0.4, marker='_')
        #plt.scatter(Xp[:,0], Xp[:,1], color='green', marker='+')
        if save:
            if name is None:
                name = 'gammaknn-boundary-%d-%d-%d'%(k, n_p, ratio)
            mysavefig(name+'.svg', **kwargs)
            mysavefig(name+'.png', **kwargs)
        if show:
            plt.show()

# %%
#%time separated(1, 100, 100, save=True, ne=ne) # works in 50s

# %%
for k in [1]:
    for n_p in [10, 100]:
        for ratio in [10, 100]:
            separated(k, n_p, ratio, save=True, final=True)






# %% TEST numexpr reduction
S = 300
a = np.random.uniform(-1, 1, (1000,1,1))
b = np.random.uniform(-1, 1, (1,2000,1))
c = np.random.uniform(-1, 1, (1,1,S))

%time np.shape(a*b*c) # 3.4 for 300, memory error at 3000

# %% one by one
%time np.min(a*b*c) # 3.93 for 300, err for 3000
%time ne.evaluate('min(a*b*c)') # 4.07 for 200, 37 for 3000
%time ne.evaluate('min(a*b*c)')
%time np.min(a*b*c)






#


# %%
