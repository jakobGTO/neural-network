import pickle
import matplotlib.pyplot as plt
import numpy as np

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def LoadBatch(filename):
    with open('Dataset/'+filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def readData():
    batch1 = LoadBatch("data_batch_1")
    batch2 = LoadBatch("data_batch_2")
    batch3 = LoadBatch("test_batch")


    X_train = batch1[b'data'].astype('float32')
    X_vali = batch2[b'data'].astype('float32')
    X_test = batch3[b'data'].astype('float32')

    y_train = batch1[b'labels']
    y_vali = batch2[b'labels']
    y_test = batch3[b'labels']

    x_mean = np.mean(X_train,axis=0)
    x_std = np.std(X_train,axis=0)

    X_train = (X_train - x_mean) / x_std
    X_vali = (X_vali - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std

    y_train_OH = np.eye(10)[y_train].T
    y_vali_OH = np.eye(10)[y_vali].T
    y_test_OH = np.eye(10)[y_test].T

    return X_train.T, X_test.T, X_vali.T, y_train, y_test, y_vali, y_train_OH, y_test_OH, y_vali_OH

def readDataLarge():
    batch1 = LoadBatch("data_batch_1")
    batch2 = LoadBatch("data_batch_2")
    batch3 = LoadBatch("data_batch_3")
    batch4 = LoadBatch("data_batch_4")
    batch5 = LoadBatch("data_batch_5")
    testbatch = LoadBatch("test_batch")

    X = np.concatenate((batch1[b'data'],batch2[b'data'],batch3[b'data'],batch4[b'data'],batch5[b'data']),0).astype(np.float32)
    y = np.concatenate((batch1[b'labels'],batch2[b'labels'],batch3[b'labels'],batch4[b'labels'],batch5[b'labels']),0)

    X_train, X_vali, y_train, y_vali = train_test_split(X,y, test_size=0.1)
    X_test = testbatch[b'data']
    y_test = testbatch[b'labels']

    x_mean = np.mean(X_train,axis=0)
    x_std = np.std(X_train,axis=0)

    X_train = (X_train - x_mean) / x_std
    X_vali = (X_vali - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std

    y_train_OH = np.eye(10)[y_train].T
    y_vali_OH = np.eye(10)[y_vali].T
    y_test_OH = np.eye(10)[y_test].T

    return X_train.T, X_test.T, X_vali.T, y_train, y_test, y_vali, y_train_OH, y_test_OH, y_vali_OH

def initializeParams(X_train,y_train,l,sig,batchnorm):
    # Using He initialization
    d = X_train.shape[0]

    W = []
    b = []
    gamma = []
    beta = []
    if sig is None:
        # Input layer
        W.append(np.random.normal(0,np.sqrt(2/d),size=(l[0],d)))
        b.append(np.zeros((l[0],1)))

        # Hidden + output layer
        for i in range(1,len(l)):
            W.append(np.random.normal(0,np.sqrt(2/W[-1].shape[0]),size=(l[i],W[-1].shape[0])))
            b.append(np.zeros((l[i],1)))
    else:
        # Input layer
        W.append(np.random.normal(0,sig,size=(l[0],d)))
        b.append(np.zeros((l[0],1)))

        # Hidden + output layer
        for i in range(1,len(l)):
            W.append(np.random.normal(0,sig,size=(l[i],W[-1].shape[0])))
            b.append(np.zeros((l[i],1)))


    if batchnorm:
        for num_nodes in l[0:len(l)-1]:
            gamma.append(np.ones((num_nodes,1)))
            beta.append(np.zeros((num_nodes,1)))

    return W,b,gamma,beta

def evaluateClassifier(X_train,W,b,gamma,beta,mu,sigma2,batchnorm):
    s = []
    X = []
    shat = []
    mu_l = []
    sigma2_l = []

    X.append(np.copy(X_train))

    if not batchnorm:
        # Input + Hidden layers
        for l in range(len(W) - 1):
            s.append(np.matmul(W[l], X[-1]) + b[l])
            X.append(np.maximum(0, s[-1]))

        # Output layer
        s.append(np.matmul(W[-1], X[-1]) + b[-1])
        X.append(softmax(s[-1]))
        
        return X, s
    else:
        # Input + Hidden layers
        for l in range(len(W) - 1):
            s.append(np.matmul(W[l], X[-1]) + b[l])
            if mu is None and sigma2 is None:
                mu_l.append(np.mean(s[-1],axis = 1))
                sigma2_l.append(np.var(s[-1],axis = 1))
            else:
                mu_l.append(mu[l])
                sigma2_l.append(sigma2[l])
        
            shat.append(np.matmul(np.diag(np.power(sigma2_l[-1] + 1e-100, -1/2)),s[-1] - np.expand_dims(mu_l[-1], axis=1)))
            stilde = gamma[l] * shat[-1] + beta[l]
            X.append(np.maximum(0,stilde))
        
        #Output layer
        s.append(np.matmul(W[-1], X[-1]) + b[-1])
        X.append(softmax(s[-1]))

        return X, s, shat, mu_l, sigma2_l

def computeCost(X_train, y_train, W, b, gamma, beta, mu, sigma2,_lambda, batchnorm):
    n = X_train.shape[1]

    p = evaluateClassifier(X_train,W, b, gamma, beta, mu, sigma2, batchnorm)[0][-1]

    lc = -np.log(np.sum(y_train*p, axis=0))
    
    l = (1/n) * np.sum(lc)
    
    j = _lambda * np.sum([np.sum(np.square(w)) for w in W])
    J = j + l
    
    return J,l

def computeAccuracy(X_test, y_test, W, b, gamma, beta, mu, sigma2):
    k = len(y_test)

    p = evaluateClassifier(X_test,W,b, gamma, beta, mu, sigma2, batchnorm)[0][-1]

    k_star = np.argmax(p, axis =0 ) 

    classified = np.sum(k_star == y_test)
    accuracy = classified/k

    return accuracy

def BatchNormBackPass(G, s, mu, v):
    n = G.shape[1]

    sigma1 = np.expand_dims(np.power(v + 1e-100, -0.5).T, axis=1)
    sigma2 = np.expand_dims(np.power(v + 1e-100, -1.5).T, axis=1)
    g1 = G * sigma1
    g2 = G * sigma2
    D = s - np.expand_dims(mu,axis=1)
    c = np.expand_dims(np.matmul(g2*D, np.ones(n)), axis=1)
    Gbatch = g1 - (1/n) * np.expand_dims(np.matmul(g1, np.ones(n)), axis=1) - (1/n) * D * c

    return Gbatch

def computeGradients(X, y_train, W, b, gamma, beta, p, s, shat, mu, sigma2, _lambda, batchnorm):
    n = X[0].shape[1]

    g = (p-y_train)

    gradW = []
    gradb = []
    gradGamma = []
    gradBeta = []

    if not batchnorm:
        # Start propogating from last layer
        gradW.append((1/n) * np.matmul(g,X[-2].T) + (2*_lambda*W[-1]))
        gradb.append(np.expand_dims((1/n) * np.matmul(g, np.ones(n)),axis=1))
        g = np.matmul(W[-1].T,g)
        h = np.where(X[-2] > 0, 1, 0)
        g = np.multiply(g, h)

        # Prop through rest of the layers
        for i in reversed(range(len(W)-1)):
            gradW.append((1/n) * np.matmul(g,X[i].T) + (2*_lambda*W[i]))
            gradb.append(np.expand_dims((1/n) * np.matmul(g, np.ones(n)),axis=1))
            g = np.matmul(W[i].T,g)
            h = np.where(X[i] > 0, 1, 0)
            g = np.multiply(g, h)

        # Reverse to normal order
        gradW.reverse()
        gradb.reverse()
        return gradW, gradb
    else:
        # Start prpopgating from last layer
        gradW.append((1/n) * np.matmul(g,X[-2].T) + (2*_lambda*W[-1]))
        gradb.append(np.expand_dims((1/n) * np.matmul(g, np.ones(n)),axis=1))
        g = np.matmul(W[-1].T,g)
        h = np.where(X[-2] > 0, 1, 0)
        g = np.multiply(g, h)

        # Prop through rest of the layers
        for i in reversed(range(len(W)-1)):
            gradGamma.append(np.expand_dims((1/n) * np.matmul(g*shat[i], np.ones(n)),axis=1))
            gradBeta.append(np.expand_dims((1/n) * np.matmul(g, np.ones(n)),axis=1))
            g = g * gamma[i]
            g = BatchNormBackPass(g, s[i], mu[i], sigma2[i])

            gradW.append((1/n) * np.matmul(g,X[i].T) + (2*_lambda*W[i]))
            gradb.append(np.expand_dims((1/n) * np.matmul(g, np.ones(n)),axis=1))

            if i >= 1:
                g = np.matmul(W[i].T, g)
                h = np.where(X[i] > 0, 1, 0)
                g = np.multiply(g, h)

        gradW.reverse()
        gradb.reverse()
        gradGamma.reverse()
        gradBeta.reverse()

        return gradW, gradb, gradGamma, gradBeta

    
def computeGradsNum(X, y, W, b, gamma, beta, _lambda, h, batchnorm):
    gradW = []
    gradb = []
    gradGamma = []
    gradBeta = []

    for j in range(len(b)):
        gradb.append(np.zeros(b[j].shape))
        for i in range(gradb[-1].shape[0]):
            for k in range(gradb[-1].shape[1]):
                b_try = []
                
                for bias in b:
                    b_try.append(np.copy(bias))

                b_try[j][i, k] = b_try[j][i, k] - h
                c1,_ = computeCost(X, y, W, b_try, gamma, beta, None, None, _lambda, batchnorm)
                b_try = []

                for bias in b:
                    b_try.append(np.copy(bias))
                
                b_try[j][i, k] = b_try[j][i, k] + h
                c2,_ = computeCost(X, y, W, b_try, gamma, beta, None, None, _lambda, batchnorm)
                gradb[j][i, k] = (c2 - c1) / (2 * h)

    for j in range(len(W)):
        gradW.append(np.zeros(W[j].shape))
        for i in range(gradW[-1].shape[0]):
            for k in range(gradW[-1].shape[1]):
                w_try = []

                for weights in W:
                    w_try.append(np.copy(weights))
                
                w_try[j][i, k] = w_try[j][i, k] - h
                c1,_ = computeCost(X, y, w_try, b, gamma, beta, None, None, _lambda, batchnorm)
                w_try = []

                for weights in W:
                    w_try.append(np.copy(weights))
                
                w_try[j][i, k] = w_try[j][i, k] + h
                c2,_ = computeCost(X, y, w_try, b, gamma, beta, None, None, _lambda, batchnorm)
                gradW[j][i, k] = (c2 - c1) / (2 * h)

    for j in range(len(gamma)):
        gradGamma.append(np.zeros(gamma[j].shape))
        for i in range(gradGamma[-1].shape[0]):
            for k in range(gradGamma[-1].shape[1]):
                g_try = []

                for gammas in gamma:
                    g_try.append(np.copy(gammas))

                g_try[j][i,k] = g_try[j][i,k] - h
                c1,_ = computeCost(X, y, W, b, g_try, beta, None, None, _lambda, batchnorm)
                g_try = []

                for gammas in gamma:
                    g_try.append(np.copy(gammas))

                g_try[j][i,k] = g_try[j][i,k] + h
                c2,_ = computeCost(X, y, W, b, g_try, beta, None, None, _lambda, batchnorm)
                gradGamma[j][i, k] = (c2 - c1) / (2 * h)

    for j in range(len(beta)):
        gradBeta.append(np.zeros(beta[j].shape))
        for i in range(gradBeta[-1].shape[0]):
            for k in range(gradBeta[-1].shape[1]):
                beta_try = []

                for betas in beta:
                    beta_try.append(np.copy(betas))

                beta_try[j][i,k] = beta_try[j][i,k] - h
                c1,_ = computeCost(X, y, W, b, gamma, beta_try, None, None, _lambda, batchnorm)
                beta_try = []

                for betas in beta:
                    beta_try.append(np.copy(betas))

                beta_try[j][i,k] = beta_try[j][i,k] + h
                c2,_ = computeCost(X, y, W, b, gamma, beta_try, None, None, _lambda, batchnorm)
                gradBeta[j][i, k] = (c2 - c1) / (2 * h)
                
    return gradW, gradb, gradGamma, gradBeta

def compareGrads(X_train, y_train_OH, batchnorm):
    X_train = X_train[0:20,0:5000]
    y_train_OH = y_train_OH[:,0:5000]

    layers = [20,10]
    layers2 = [20,20,10]
    layers3 = [20,20,20,10]

    sig = None

    W,b, gamma, beta = initializeParams(X_train, y_train_OH, layers, sig, batchnorm)
    p = evaluateClassifier(X_train,W,b,gamma,beta,None,None,batchnorm)[0][-1]
    X, s, shat, mu, sigma2 = evaluateClassifier(X_train,W,b,gamma,beta,None,None,batchnorm)

    gradW, gradb, gradGamma, gradBeta = computeGradients(X, y_train_OH, W, b, gamma, beta, p, s, shat, mu, sigma2, _lambda=0, batchnorm=batchnorm)
    numgradW,numgradb,numgradGamma,numgradBeta = computeGradsNum(X_train, y_train_OH, W, b, gamma, beta,_lambda=0, h=1e-4, batchnorm=batchnorm)

    for i in range(len(gradW)):
        print('W_',str(i),"is sufficiently similar: ",np.mean(abs(gradW[i] - numgradW[i])) < 1e-5)
        print('b_',str(i),"is sufficiently similar: ",np.mean(abs(gradb[i] - numgradb[i])) < 1e-5)


    for j in range(len(gradGamma)):
        print('gamma_',str(j),"is sufficiently similar: ",np.mean(abs(gradGamma[j] - numgradGamma[j])) < 1e-5)
        print('beta_',str(j),"is sufficiently similar: ",np.mean(abs(gradBeta[j] - numgradBeta[j])) < 1e-5)



def MiniBatchGD(X_train, y_train_OH, y_train, X_vali, y_vali_OH, y_vali, n_batch, eta_min, eta_max, n_s, n_epochs, W, b, gamma, beta, alpha,_lambda, batchnorm):
    train_cost_arr = []
    train_loss_arr = []

    vali_cost_arr = []
    vali_loss_arr = []

    train_acc_arr = []
    vali_acc_arr = []

    eta = eta_min
    l = 0.0
    t = 0.0
    
    for i in range(n_epochs):
        for j in range(int(X_train.shape[1]/n_batch)):
            j_start = (j-1)*n_batch
            j_end = j*n_batch
            inds = range(j_start,j_end)
            Xbatch = X_train[:,inds]
            Ybatch = y_train_OH[:,inds]

            W_old = W
            b_old = b
            gamma_old = gamma
            beta_old = beta
            
            if not batchnorm:
                shat = None
                mu = None
                sigma2 = None
                X, s = evaluateClassifier(Xbatch, W_old, b_old, gamma_old, beta_old, mu, sigma2, batchnorm)
                gradW, gradb = computeGradients(X, Ybatch, W_old, b_old, gamma_old, beta_old, X[-1], s, shat, mu, sigma2, _lambda, batchnorm)
            else:
                X, s, shat, mu, sigma2 = evaluateClassifier(Xbatch, W_old, b_old, gamma_old, beta_old, None, None, batchnorm)
                gradW, gradb, gradGamma, gradBeta = computeGradients(X, Ybatch, W_old, b_old, gamma_old, beta_old, X[-1], s, shat, mu, sigma2, _lambda, batchnorm)

                for gU in range(len(gamma_old)):
                    gamma[gU] = gamma_old[gU] - eta * gradGamma[gU]

                for betaU in range(len(beta_old)):
                    beta[betaU] = beta_old[betaU] - eta * gradBeta[betaU]

                if j == 0:
                    mu_avg = mu
                    v_avg = sigma2
                else:
                    for l in range(len(mu_avg)):
                        mu_avg[l] = alpha * mu_avg[l] + (1 - alpha) * mu[l]
                        v_avg[l] = alpha * v_avg[l] + (1 - alpha) * sigma2[l]

            for wU in range(len(W_old)):
                W[wU] = W_old[wU] - eta * gradW[wU]

            for bU in range(len(b_old)):
                b[bU] = b_old[bU] - eta * gradb[bU]
            
            l = int(t / (2 * n_s))
            t += 1
            if (2*l*n_s <= t <= (2*l + 1)*n_s):
                eta = eta_min + ((t-2*l*n_s) / n_s)*(eta_max-eta_min)
            else:
                eta = eta_max - ((t - ((2*l)+1)*n_s) / n_s)*(eta_max-eta_min)

        if batchnorm:
            train_cost,train_loss = computeCost(X_train, y_train_OH, W, b, gamma, beta, mu_avg, v_avg, _lambda, batchnorm)
            vali_cost,vali_loss = computeCost(X_vali, y_vali_OH, W, b, gamma, beta, mu_avg, v_avg, _lambda, batchnorm)
            
            train_acc = computeAccuracy(X_train, y_train, W, b, gamma, beta, mu_avg, v_avg)
            vali_acc = computeAccuracy(X_vali, y_vali, W, b, gamma, beta, mu_avg, v_avg)
        else:
            train_cost,train_loss = computeCost(X_train, y_train_OH, W, b, gamma, beta, None, None, _lambda, batchnorm)
            vali_cost,vali_loss = computeCost(X_vali, y_vali_OH, W, b, gamma, beta, None, None, _lambda, batchnorm)
            
            train_acc = computeAccuracy(X_train, y_train, W, b, gamma, beta, None, None)
            vali_acc = computeAccuracy(X_vali, y_vali, W, b, gamma, beta, None, None)

        print('')
        print('Epoch:', i+1)
        print('')
        print('Training cost: ', train_cost)
        print('Validation cost: ', vali_cost)
        print('')
        print('Training loss: ', train_loss)
        print('Validation loss: ', vali_loss)
        print('')
        print('Training acc: ', train_acc)
        print('Validation acc: ', vali_acc)

        train_cost_arr.append(train_cost)
        train_loss_arr.append(train_loss)

        vali_cost_arr.append(vali_cost)
        vali_loss_arr.append(vali_loss)

        train_acc_arr.append(train_acc)
        vali_acc_arr.append(vali_acc)

        # Random shuffling
        
        rng = np.arange(X_train.shape[1])
        np.random.shuffle(rng)
        X_train = X_train[:,rng]
        y_train_OH = y_train_OH[:,rng]
        y_train = np.array(y_train)[rng]
        
        
    if batchnorm:        
        return W,b,gamma,beta,mu_avg,v_avg,train_cost_arr,train_loss_arr,vali_cost_arr,vali_loss_arr,train_acc_arr,vali_acc_arr
    else:
        return W,b,gamma,beta,None,None,train_cost_arr,train_loss_arr,vali_cost_arr,vali_loss_arr,train_acc_arr,vali_acc_arr

def plot_all(train_cost_arr,train_loss_arr,vali_cost_arr,vali_loss_arr,train_acc_arr,vali_acc_arr):
    plt.plot(train_cost_arr,label='Train cost')
    plt.plot(vali_cost_arr,label='Vali cost')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.legend()
    plt.title('eta_min: ' + str(eta_min) + ', eta_max: ' + str(eta_max) + ', n_batch: ' + str(n_batch) + ', n_epochs: ' + str(n_epochs) + ', lambda: ' + str(_lambda))
    plt.show()

    plt.plot(train_loss_arr,label='Train loss')
    plt.plot(vali_loss_arr,label='Vali loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('eta_min: ' + str(eta_min) + ', eta_max: ' + str(eta_max) + ', n_batch: ' + str(n_batch) + ', n_epochs: ' + str(n_epochs) + ', lambda: ' + str(_lambda))
    plt.show()

    plt.plot(train_acc_arr,label='Train acc')
    plt.plot(vali_acc_arr,label='Vali acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.title('eta_min: ' + str(eta_min) + ', eta_max: ' + str(eta_max) + ', n_batch: ' + str(n_batch) + ', n_epochs: ' + str(n_epochs) + ', lambda: ' + str(_lambda))
    plt.show()

if __name__ == '__main__':
    np.random.seed(970922)

    X_train, X_test, X_vali, y_train, y_test, y_vali, y_train_OH, y_test_OH, y_vali_OH = readDataLarge()
    
    #3-layer and 9-layer net testing with and without BN
    
    #50 hidden nodes, 10 output nodes for k = 10
    eta_min = 1e-5
    eta_max = 1e-1
    n_batch = 100
    n_s = 5 * np.floor(X_train.shape[1] / n_batch)
    n_epochs = 20
    _lambda = 0.005
    alpha = 0.75
    batchnorm = True
    sig = None

    #layers = [50,30,20,20,10,10,10,10,10]
    layers = [50,50,10]

    W_old,b_old,gamma_old,beta_old = initializeParams(X_train, y_train_OH, layers, sig, batchnorm)
    W,b,gamma,beta,mu_avg,v_avg,train_cost_arr,train_loss_arr,vali_cost_arr,vali_loss_arr,train_acc_arr,vali_acc_arr = MiniBatchGD(X_train, y_train_OH, y_train, X_vali, y_vali_OH, y_vali, n_batch, eta_min, eta_max, n_s, n_epochs, W_old, b_old, gamma_old, beta_old, alpha, _lambda, batchnorm)

    acc = computeAccuracy(X_test, y_test, W, b, gamma, beta, mu_avg, v_avg)
    print('\nAccuracy on the test set:', acc)

    plot_all(train_cost_arr,train_loss_arr,vali_cost_arr,vali_loss_arr,train_acc_arr,vali_acc_arr)
    
