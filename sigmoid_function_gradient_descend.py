import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def equation(x1,x2,x3):
    return 5*x1+2*x2-2*x3


def sigma(lsx,sfunc):
    result = []
    for x in lsx:
     result.append((1/(1+np.exp((-x*sfunc[0])+sfunc[1]))))
    return result

def margin(data, labels, th, th0):
    return ((np.dot(th.T, data) + th0) * labels) / np.linalg.norm(th)


def hinge_los(data,labels,th,th0,mref):
    hinge = 0
    hinge_l = []
    marg = margin(data,labels,th,th0)
    print(type(marg))
    print(np.shape(marg))
    for mar in marg.T:
        if mar<mref:
            hinge+=(1-mar/mref)
            hinge_l.append((1-mar/mref))
        elif mar > mref:
            hinge_l.append(0)



def yref(data,labels,th,th0):
    yrefe = 1.41/2
    return np.array([1,1,1]) - margin(data,labels,th,th0)/yrefe


def svm(data,labels,th,th0):
    return ((np.dot(th.T,data)+th0)*labels)


def step_size_fn(t):
    return 1/(t+1)
def gd(f, df, x0, step_size_fn, max_iter):
    x = x0
    t = 0
    fs = []
    xs = []
    while t<max_iter:
        fs.append(f(x))
        xs.append(x)
        x = x - step_size_fn(t)*df(x)
        t+=1
    fs.append(f(x))
    xs.append(x)
    return x,fs,xs
def gd(f, df, x0, step_size_fn, max_iter): # many iterations many steps!!
    prev_x = x0
    fs = []; xs = []
    for i in range(max_iter):
        prev_f, prev_grad = f(prev_x), df(prev_x)
        fs.append(prev_f); xs.append(prev_x)
        if i == max_iter-1:
            return prev_x, fs, xs # x the value at final step. what does it even return, wll, for many steps, it returns
        # all the gradients and the x s ? so I pick the smallest gradient? or what?
        step = step_size_fn(i)
        prev_x = prev_x - step * prev_grad

def package_ans(gd_vals):
    x, fs, xs = gd_vals
    return [x.tolist(), [fs[0], fs[-1]], [xs[0].tolist(), xs[-1].tolist()]]

def num_grad(f, delta=0.001):
    def df(x):
        g = np.zeros(x.shape)
        for i in range(x.shape[0]):
            xi = x[i,0]
            x[i,0] = xi - delta
            fxm = f(x)
            x[i,0] = xi + delta
            fxp = f(x)
            x[i,0] = xi
            g[i,0] = (fxp - fxm)/(2*delta)
        return g
    return df

def minimize(f, x0, step_size_fn, max_iter): # return local minimum
    df= num_grad(f, delta=0.001)
    x,fs,xs = gd(f, df, x0, step_size_fn, max_iter)
    return x



def hinge(v):
    print("v ",v)
    return np.where(v<1,1-v,0) # where v smaller than 1 becomes 1-v otherwise it is 0

# v also has to be defined?
# x is dxn, y is 1xn, th is dx1, th0 is 1x1
def hinge_loss(x, y, th, th0):
    v = y*(th.T@x+th0)
    return hinge(v)

# x is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
def svm_obj(x, y, th, th0, lam):
    # go through all how?
    return np.average(hinge_loss(x,y,th,th0)) + lam*(np.linalg.norm(th)**2)


# gd(f, df, x0, step_size_fn, max_iter) # many steps
# num_grad(f, delta=0.001) # for each point in one step

# Returns the gradient of hinge(v) with respect to v.
def d_hinge(v):
    return np.where(v>1,0,-1)

def hinge_loss(x, y, th, th0):
    v = y*(th.T@x+th0)
    return hinge(v)

# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th
def d_hinge_loss_th(x, y, th, th0):
    return  d_hinge(y*(np.dot(th.T, x) + th0))* y * x # basically the derivative of hinge_loss

# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th0
def d_hinge_loss_th0(x, y, th, th0):
    return d_hinge(y*(np.dot(th.T, x) + th0)) * y
#############################################
# x is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
def svm_obj(x, y, th, th0, lam):
    # go through all how?
    return np.average(hinge_loss(x,y,th,th0)) + lam*(np.linalg.norm(th)**2)

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th
def d_svm_obj_th(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th(x, y, th, th0), axis = 1, keepdims = True) + lam * 2 * th
# Returns the gradient of svm_obj(x, y, th, th0) with respect to th0
def d_svm_obj_th0(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th0(x, y, th, th0), axis = 1, keepdims = True)

# Returns the full gradient as a single vector (which includes both th, th0)
def svm_obj_grad(X, y, th, th0, lam):
    grad_th = d_svm_obj_th(X, y, th, th0, lam)
    grad_th0 = d_svm_obj_th0(X, y, th, th0, lam)
    return np.vstack([grad_th, grad_th0])

def batch_svm_min(data, labels, lam):
    def svm_min_step_size_fn(i):
       return 2/(i+1)**0.5
    init = np.zeros((data.shape[0] + 1, 1))
    def f(th):
      return svm_obj(data, labels, th[:-1, :], th[-1:,:], lam)
    def df(th):
      return svm_obj_grad(data, labels, th[:-1, :], th[-1:,:], lam)
    x, fs, xs = gd(f, df, init, svm_min_step_size_fn, 10)
    return x, fs, xs