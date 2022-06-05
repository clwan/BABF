import torch
import sys
import numpy as np
import torch.nn as nn
import torch.optim as optim


import numpy as np
import multiprocessing as mp
from collections import Counter
from bisect import bisect
from functools import partial
from random import sample


def Prob_cdf(prob):
    prob_frequency=Counter(prob)
    prob_value=list(prob_frequency.keys())
    prob_value.sort()
    prob_sample=[x*prob_frequency[x] for x in prob_value]
    prob_sample=prob_sample/sum(prob_sample)
    prob_cdf=prob_sample*0
    prob_cdf[0]=prob_sample[0]
    for i in range(1,len(prob_sample)):
        prob_cdf[i]=prob_cdf[i-1]+prob_sample[i]
    return prob_value,prob_cdf


def calculate_weight(x,prob,prob_value,prob_cdf):
    prob_test=[prob[i] for i,y in enumerate(x) if y>0]
    prob_test.sort()
    weight=0
    n=len(prob_test)
    if n>1:
        for i in range(n-1):
            cdf_v=(i+1)/n
            pos=bisect(prob_cdf,cdf_v)
            prob_v=prob_value[pos]
            if prob_v<1:
                weight_v=(prob_v-prob_test[i])/(1-prob_v)
                if weight_v<0:
                    weight=weight-weight_v
    return weight 


def BIND_prob(MAT,axis,core=None):
    prob=np.mean(MAT,1-axis)
    if any(i==0 for i in prob):
        raise Exception('matrix has all zero line on axis {}'.format(str(1-axis)))
    prob_value,prob_cdf=Prob_cdf(prob)
    if axis==0:
        MAT_tolist=MAT.T.tolist()
        if core is None:
            weight=[calculate_weight(x,prob,prob_value,prob_cdf) for x in MAT_tolist]
        else:
            with mp.Pool(core,maxtasksperchild=100) as pool:
                weight=pool.imap(partial(calculate_weight,prob=prob,prob_value=prob_value,prob_cdf=prob_cdf), MAT_tolist,chunksize=100)
        weight=[round(x) for x in weight]
        SUM=[sum(x) for x in MAT_tolist]
        prob_new=[(SUM[i]-weight[i])/(len(MAT_tolist[0])-weight[i]) for i in range(len(weight))]
    else:
        MAT_tolist=MAT.tolist()
        if core is None:
            weight=[calculate_weight(x,prob,prob_value,prob_cdf) for x in MAT_tolist]
        else:
            with mp.Pool(core,maxtasksperchild=100) as pool:
                weight=pool.imap(partial(calculate_weight,prob=prob,prob_value=prob_value,prob_cdf=prob_cdf), MAT.tolist(),chunksize=100)       
        weight=[round(x) for x in weight]
        SUM=[sum(x) for x in MAT_tolist]
        prob_new=[(SUM[i]-weight[i])/(len(MAT_tolist[0])-weight[i]) for i in range(len(weight))]    
    return prob_new



def get_random_matrices(M,N,m,n,K,prob_b=[0.0,0.6],prob_p=0.95,prob_n=0.05):
    """
    M, N: matrix size
    m, n: pattern size
    K: pattern number
    prob_b: the max value of background probability that follows uniform distribution # default 0.6
    prob_p: the bernoulli probability of pattern default 0.95
    prob_n: the bernoulli probability of noise default 0.05
    """
    X,Y=[],[]
    for i in range(K):
        temp=np.array(sample(range(M),m))
        X.append(temp)
        temp=np.array(sample(range(N),n))
        Y.append(temp)

    O=np.zeros((M,N))
    X1=np.zeros((M,K))
    Y1=np.zeros((N,K))
    for i in range(K):
        temp=np.random.binomial(1, prob_p, m*n)
        O[np.ix_(X[i],Y[i])]=temp.reshape(m,n)
        X1[np.ix_(X[i],np.array([i]))]=1
        Y1[np.ix_(Y[i],np.array([i]))]=1

    # Z=(X1.dot(Y1.T) > 0).astype(float)

    # background
    b_x=np.random.uniform(prob_b[0],prob_b[1],M)
    b_y=np.random.uniform(prob_b[0],prob_b[1],N)

    B_X=[]
    for i in range(M):
        temp=np.random.binomial(1,b_x[i], N)
        B_X.append(temp)

    B_Y=[]
    for i in range(N):
        temp=np.random.binomial(1,b_y[i], M)
        B_Y.append(temp)

    B_X=np.array(B_X)
    B_Y=np.array(B_Y)
    B=B_X*B_Y.T
    O=(O+B>0).astype(int)

    # noise
    flip = np.random.rand(M,N) < prob_n
    O[flip] = 1 - O[flip]

    # mask = np.random.rand(M,N) < prob_o   
    mats = {'X':X1, 'Y':Y1,'O':O, 'b_x':b_x, 'b_y':b_y}
    return mats







class Bias_BIND(object):  
    def __init__(self,
                 O, #observed matrix (only the parts indicated by mask will be used)
                 K, #hidden dim
                 mask= None,#boolean matrix the same size as O                 
                 tol = 1e-4,#tolerance for message updates
                 lr1 = .1, #damping parameter for general
                 lr2 =.01, # learning rate to call bias
                 m1 = 100, # maximum number for message passing update
                 m2 = 10, # maximum number for bias update
                 max_iter = 1000, #maximum number of whole update
                 verbose = False,
                 p_x_1 = .5, #the prior probability of x=1. For regularization use small or large values in [0,1]
                 p_y_1 = .5, #the prior probability of y=1. For regularization use small or large values in [0,1]
                 #note that when p_x and p_y are uniform the MAP assignment is not sensitive
                 #to the following values, assuming they are the same and above .5
                 p1 = .99, #the model of the noisy channel: probability of observing 1 for the input of 1
                 p0 = .01 # probability of observing 1 for the input of 0
                ):
        
        assert(p_x_1 < 1 and p_x_1 > 0)
        assert(p_y_1 < 1 and p_y_1 > 0)
        assert(p1 > .5 and p1 < 1)               
        
        self.O = O.astype(int)
        self.M,self.N = O.shape
        self.K = K
        self.verbose = verbose


        assert(self.K < min(self.M,self.N))
        if mask is not None:
            assert(mask.shape[0] == self.M and mask.shape[1] == self.N)
            self.mask = mask.astype(bool)
        else:
            self.mask = np.ones(self.O.shape, dtype=bool)
            
        self.learning_rate = lr1
        self.max_iter = max_iter
        self.tol = tol
        self.m1=m1
        self.m2=m2
        self.num_edges = np.sum(self.mask)        

        self.update_adj_list()
        
        # will be used frequently
        self.pos_edges = np.nonzero(O[mask])[0]
        self.neg_edges = np.nonzero(1 - O[mask])[0]
        self.range_edges = np.arange(self.num_edges)
        self.cx = np.log(p_x_1) - np.log(1 - p_x_1)
        self.cy = np.log(p_y_1) - np.log(1 - p_y_1)

        self.p1=p1
        self.p0=p0

        # self.b_x=np.random.rand(self.M)
        # self.b_y=np.random.rand(self.N)

        self.b_x=np.ones(self.M)*p0
        self.b_y=np.ones(self.N)*p0

    
    def init_msgs_n_marginals(self):
        self.marg_x = np.zeros((self.M, self.K))
        self.marg_y = np.zeros((self.N, self.K))
        self.in_x = np.zeros((self.num_edges, self.K)) #message going towards variable X: phi in the papger
        self.new_in_x = np.zeros((self.num_edges, self.K)) #the new one
        
        self.out_x = np.log((np.random.rand(self.num_edges, self.K)))#/self.M #message leaving variable x: phi_hat in the paper 
        self.in_y = np.zeros((self.num_edges, self.K)) #message leaving variable y: psi in the paper
        self.new_in_y = np.zeros((self.num_edges, self.K))
        self.out_y = np.log(np.random.rand(self.num_edges, self.K))#/self.N #psi_hat in the paper
        self.in_z = np.zeros((self.num_edges, self.K)) #gamma in the paper
        self.out_z = np.zeros((self.num_edges, self.K)) #gamma_hat in the paper
        
        
    def update_adj_list(self):
        ''' nbM: list of indices of nonzeros organized in rows
        nbM: list of indices of nonzeros organized in columns
        '''
        
        Mnz,Nnz = np.nonzero(self.mask)
        M = self.M
        N = self.N
        nbM = [[] for i in range(M)] 
        nbN = [[] for i in range(N)]

        for z in range(len(Mnz)):
            nbN[Nnz[z]].append(z)
            nbM[Mnz[z]].append(z)

        for i in range(M):
            nbM[i] = np.array(nbM[i], dtype=int)
        for i in range(N):
            nbN[i] = np.array(nbN[i], dtype=int)
            
        self.rows = nbM
        self.cols = nbN



    def update_min_sum(self):
        self.in_z = np.minimum(np.minimum(self.out_x + self.out_y, self.out_x), self.out_y) #gamma update in the paper
        
        inz_pos = np.maximum(0.,self.in_z) # calculate it now, because we're chaning inz
        #find the second larges element along the 1st axis (there's also a 0nd! axis)
        inz_max_ind = np.argmax(self.in_z, axis=1)
        inz_max = np.maximum(-self.in_z[self.range_edges, inz_max_ind],0)
        self.in_z[self.range_edges, inz_max_ind] = -np.inf
        inz_max_sec = np.maximum(-np.max(self.in_z, axis=1),0) # update for gamma_hat in the paper
        sum_val = np.sum(inz_pos, axis=1)
        #penalties/rewards for confoming with observations
        sum_val[self.pos_edges] += self.co1[self.pos_edges]
        sum_val[self.neg_edges] += self.co0[self.neg_edges]
        
        tmp_inz_max = inz_max.copy()
        inz_pos =  sum_val[:, np.newaxis] - inz_pos
        
        for k in range(self.K):
            self_max_ind = np.nonzero(inz_max_ind == k)[0]#find the indices where the max incoming message is from k
            tmp_inz_max[self_max_ind] = inz_max_sec.take(self_max_ind)#replace the value of the max with the second largest value
            self.out_z[:, k] = np.minimum( tmp_inz_max, inz_pos[:,k])#see the update for gamma_hat
            tmp_inz_max[self_max_ind] = inz_max.take(self_max_ind)#fix tmp_iz_max for the next iter

        # update in_x and in_y: phi_hat and psi_hat in the paper
        self.new_in_x = np.maximum(self.out_z + self.out_y, 0) - np.maximum(self.out_y,0)
        self.new_in_y = np.maximum(self.out_z + self.out_x, 0) - np.maximum(self.out_x,0)



    def update_margs(self):
        #updates for phi and psi
        for m in range(self.M):
            self.marg_x[m,:] = np.sum(self.in_x.take(self.rows[m],axis=0), axis=0) + self.cx
            self.out_x[self.rows[m], :] = -self.in_x.take(self.rows[m],axis=0) + self.marg_x[m,:]

        for n in range(self.N):
            self.marg_y[n, :] = np.sum(self.in_y.take(self.cols[n], axis=0), axis=0) + self.cy
            self.out_y[self.cols[n], :] = -self.in_y.take(self.cols[n], axis=0) + self.marg_y[n,:]
        # self.X = (self.marg_x > 0).astype(int)
        # self.Y = (self.marg_y > 0).astype(int)
        # self.Z = (self.X.dot(self.Y.T) > 0).astype(int)


    def run_BMF(self):
        #self.init_msgs_n_marginals()
        iters = 1
        diff_msg = np.inf
        while (diff_msg > self.tol and iters <= self.m1) or iters < 5:
            self.update_min_sum()#(outX, outY, inZ, outZ, newInX, newInY, posEdges, negEdges,  opt)
            diff_msg = np.max(np.abs(self.new_in_x - self.in_x))
            self.in_x *= (1. - self.learning_rate)
            self.in_x += self.learning_rate * (self.new_in_x)
            self.in_y *= (1. - self.learning_rate)
            self.in_y += self.learning_rate * (self.new_in_y)
            self.update_margs()
            #self.update_bias()
            if self.verbose:
                print("BMF iter %d, diff:%f" %(iters, diff_msg))
            # else:
            #     print(".")
            #     sys.stdout.flush()
                         
            iters += 1
        #recover X and Y from marginals and reconstruct Z
        self.X = (self.marg_x > 0).astype(int)
        self.Y = (self.marg_y > 0).astype(int)
        self.Z = (self.X.dot(self.Y.T) > 0).astype(int)



    def calculate_bias(self):
        '''
        update bias based on the bias call model
        '''
        self.bias=np.outer(self.b_x,self.b_y)
        self.bias=np.minimum(self.p1,self.bias)
        self.bias=np.maximum(self.p0,self.bias)
        self.bias=self.bias[self.mask]
        p1_new=1-(1-self.p1)*(1-self.bias)
        self.co1=np.log(p1_new) - np.log(self.bias)
        self.co0=np.log(1-p1_new)-np.log(1-self.bias)

    def update_bias(self):
        new_mask = self.Z.astype(bool)

        SUM=np.sum(new_mask,1)
        X1=np.sum(self.b_y)
        X2=self.O.dot(self.b_y)
        for i in range(self.M):
            if SUM[i]>0:
                X1_use=X1-np.sum(self.b_y[new_mask[i,:]])
                X2_use=X2[i]-np.sum(self.b_y[new_mask[i,:]]*self.O[i,:][new_mask[i,:]])
            else:
                X1_use=X1
                X2_use=X2[i]

            val=X2_use/X1_use if X1_use>0 else 0
            val=val if val<1 else 1
            val=val if val>0 else 0
            self.b_x[i]=val

        SUM=np.sum(new_mask,0)
        X1=np.sum(self.b_x)
        X2=self.O.T.dot(self.b_x)
        for i in range(self.N):
            if SUM[i]>0:
                X1_use=X1-np.sum(self.b_x[new_mask[:,i]])
                X2_use=X2[i]-np.sum(self.b_x[new_mask[:,i]]*self.O[:,i][new_mask[:,i]])
            else:
                X1_use=X1
                X2_use=X2[i]

            val=X2_use/X1_use if X1_use>0 else 0
            val=val if val<1 else 1
            val=val if val>0 else 0
            self.b_y[i]=val


    def run_Bias(self):
        new_mask=(1-self.Z).astype(bool)
        bias=np.outer(self.b_x,self.b_y)
        loss=np.sum(np.square(self.O[new_mask]-bias[new_mask]))
        b_x=self.b_x
        b_y=self.b_y
        for iters in range(self.m2):
            self.update_bias()
            bias=np.outer(self.b_x,self.b_y)
            new_loss=np.sum(np.square(self.O[new_mask]-bias[new_mask]))
            if new_loss<loss:
                loss=new_loss
                b_x=self.b_x
                b_y=self.b_y
            if self.verbose:
                print("Bias iter %d, diff:%f" %(iters, new_loss))
            # else:
            #     print(".")
            #     sys.stdout.flush()

        self.b_x=b_x
        self.b_y=b_y

    def run(self):
        self.init_msgs_n_marginals()
        iters=1
        diff_msg=np.inf
        current_x=self.b_x
        current_in_x=self.in_x
        while (diff_msg > self.tol and iters <= self.max_iter) or iters < 10:
            self.calculate_bias()
            self.run_BMF()
            self.run_Bias()
            diff_msg=np.maximum(np.max(np.abs(self.b_x - current_x)),np.max(np.abs(self.in_x - current_in_x)))
            iters+=1
            current_x=self.b_x
            current_in_x=self.in_x
            if self.verbose:
                print("Overall iter %d, diff:%f" %(iters, diff_msg))
            # else:
            #     print(".")
            #     sys.stdout.flush()

    def loglikelihood(self,Cross_val=False):
        p_all=1-(1-np.outer(self.b_x,self.b_y))*(1-self.Z)
        p_all[p_all<self.p0]=self.p0
        p_all[p_all>self.p1]=self.p1
        llh=self.O*np.log(p_all)+(1-self.O)*np.log(1-p_all)
        if Cross_val:
            self.llh=np.sum((1-self.mask)*llh)
        else:
            self.llh=np.sum(llh)

    def model_eval(self):
        if not hasattr(self,"llh"):
            self.loglikelihood()

        n_para=(self.K+1)*(self.M+self.N)

        self.aic=2*n_para-2*self.llh
        self.bic=n_para*np.log(self.M*self.N)-2*self.llh



        # val1=log_all_pos(self.M)+log_all_pos(self.N)+2*2.865+np.log(min(self.M,self.N))
        # val2=decode_factor_matrix(self.X)+decode_factor_matrix(self.Y)
        # val3=np.log(self.M*self.N)+np.sum(np.abs())




def log_all_pos(val):
    val_all=0
    while np.log(val)>0:
        val=np.log(val)
        val_all+=val
    return val_all


def decode_factor_matrix(MAT):
    m,n=MAT.shape
    val1=np.log(m)*n
    val=0
    for i in range(n):
        vec=MAT[:,i]
        v_sum=np.sum(vec)
        val+=-v_sum*np.log(v_sum/m)-(m-v_sum)*np.log((m-v_sum)/m)
    return val+val1




















