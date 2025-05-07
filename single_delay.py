import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from scipy.special import loggamma as LG
from numpy import linalg as LA
from numpy.random import gamma as IG
from numpy.random import beta
from numpy.random import binomial as bern
from numpy.random import multivariate_normal as mvrv
from scipy.stats import dirichlet, multinomial
import utils
import math
from sklearn.linear_model import LinearRegression
from pysindy.feature_library import CustomLibrary


'''
GIBBS SAMPLING FOR MULTIPLE TIME DELAYS
'''

class Gibbs(object):
    def __init__(self, num_delays, num_variables, num_targets, num_timesteps, dt, functions, tau_alphas=None, start_indices=None, end_indices=None, 
                 alpha_v=0.5, beta_v=0.5, alpha_p=0.1, beta_p=0.1, alpha_sig=1e-4, beta_sig=1e-4, p_initial=0.1, 
                 v_initial=0.1, num_samples=500, burn_in = 250, G_threshold = 1e-250):
        
        # HYPER-PARAMETERS
        self.num_variables = num_variables
        self.num_delays = 1
        self.N = num_timesteps
        self.ns = num_targets
        self.ap = alpha_p
        self.bp = beta_p
        self.av = alpha_v
        self.bv = beta_v
        self.asig = alpha_sig
        self.bsig = beta_sig
        self.MCMC = num_samples
        self.burn_in = burn_in
        self.dt = dt
        self.G_threshold = G_threshold
        self.known_tau = False
        
        if np.any(start_indices==None) | np.any(end_indices==None):
            for t in range(self.num_delays):
                if t>0:
                    start_indices[t] = int(100*t)
                    end_indices[t] = start_indices+100
                else:
                    start_indices[t] = 50 + int(100*t)
                    end_indices[t] = start_indices+100
                    
        self.Nalphas = np.array(end_indices-start_indices)
        self.start_indices = np.array(start_indices)
        print('initial Nalphas = ', self.Nalphas)
        
        self.Library = CustomLibrary(library_functions=functions)
        print('library fit data = ', int((self.num_delays+1)*self.num_variables))
        self.Library.fit(np.zeros(int((self.num_delays+1)*self.num_variables)))
        self.nl = len(self.Library.get_feature_names())
        self.zstore = np.zeros((self.nl, self.MCMC-self.burn_in))
        
        # PARAMETER INITIALIZATION
        self.p0 = np.zeros(self.MCMC)
        self.v = np.zeros(self.MCMC)
        self.sigma = np.zeros(self.MCMC)
        self.theta = np.zeros((self.nl, self.MCMC-self.burn_in))
        self.taus = np.zeros((num_delays, self.MCMC), int)
        self.G = []
        self.G_prior = []
        for j in range(self.num_delays):
            self.G.append(np.ones(self.Nalphas[j])/self.Nalphas[j])
            self.G_prior.append(np.ones(self.Nalphas[j]))
        
        self.p0[0] = p_initial
        self.v[0] = v_initial
        self.taus[:,0] = 0
        
        
    def linreg(self, data, target):
        model = LinearRegression(fit_intercept=False)
        model.fit(data,target)
        thet = model.coef_.ravel()
        thresh = np.dot(np.abs(thet).max(), 0.1)
        latent_initial = (np.abs(thet) > thresh)
        return latent_initial
    
    
    def res_var(self, L, y):
        # Residual variance:
        beta = np.dot(LA.pinv(L), y)
        error = y - np.matmul(L, beta)
        return np.var(error)
    
    
    def Bsig_and_Bmu(self, z, L, vs, y, prior="independent"):
            # Theta: Multivariate Normal distribution
            index = np.where(z != 0)[0]
            Dr = L[:,index] 
            if prior == "independent":
                Aor = np.eye(len(index)) # independent prior
            else:
                Aor = np.dot(len(Dr), LA.inv(np.matmul(Dr.T, Dr))) # g-prior
                
            SIG_inv = np.matmul(Dr.T,Dr) + np.dot(pow(vs,-1), LA.inv(Aor))
            SIG = LA.inv(SIG_inv)
            MU = np.matmul(np.matmul(SIG,Dr.T),y).ravel()
            return MU, SIG, Aor, index, SIG_inv


    def pyzv(self, D, ztemp, vs, y, prior="independent"):
        # P(Y|zi=(0|1),z-i,vs)
        rind = np.where(ztemp != 0)[0]
        Sz = sum(ztemp)
        Dr = D[:, rind] 
        if prior == "independent":
            Aor = np.eye(len(rind)) # independent prior
        else:
            Aor = np.dot(self.N, LA.inv(np.matmul(Dr.T, Dr))) # g-prior
        
        BSIG = np.matmul(Dr.T, Dr) + np.dot(pow(vs, -1),LA.inv(Aor))
    
        (sign, logdet0) = LA.slogdet(LA.inv(Aor))
        (sign, logdet1) = LA.slogdet(LA.inv(BSIG))                                              
        
        PZ = LG(self.asig + 0.5*self.N) -0.5*self.N*np.log(2*np.pi) - 0.5*Sz*np.log(vs) \
                + self.asig*np.log(self.bsig) - LG(self.asig) + 0.5*logdet0 + 0.5*logdet1
        denom1 = np.eye(self.N) - np.matmul(np.matmul(Dr, LA.inv(BSIG)), Dr.T)
        denom = (0.5*np.matmul(np.matmul(y.T, denom1), y))
        PZ = PZ - (self.asig+0.5*self.N)*(np.log(self.bsig + denom))
        # if PZ>0:
        #     print('PZ is more than zero: ', PZ)
        return PZ


    def pyzv0(self, y):
        # P(Y|zi=0,z-i,vs)
        PZ0 = LG(self.asig + 0.5*self.N) - 0.5*self.N*np.log(2*np.pi) \
            + self.asig*np.log(self.bsig) - LG(self.asig) \
                + np.log(1) - (self.asig+0.5*self.N)*np.log(self.bsig + 0.5*np.matmul(y.T, y))
        return PZ0
    
    
    def slab(self, theta, Aor, Sz, sigma):
        # sample 'vs' from inverse Gamma:
        avvs = self.av + 0.5*Sz
        bvvs = self.bv + (np.matmul(np.matmul(theta.T, LA.inv(Aor)), theta))/(2*sigma)
        return 1/IG(avvs, 1/bvvs) # inverse gamma RVs
    
    
    def binary(self, Sz):
        # sample 'p0' from Beta distribution:
        app0 = self.ap+Sz
        bpp0 = self.bp + self.nl - Sz # Here, P=nl (no. of functions in library)
        return beta(app0, bpp0)
    
    
    def noise_sig(self, MU, BSIG, y):
        # sample 'sig^2' from inverse Gamma:
        asiggamma = self.asig+0.5*self.N
        temp = np.matmul(np.matmul(MU.T, LA.inv(BSIG)), MU)
        bsiggamma = self.bsig+0.5*(np.dot(y.T, y) - temp)
        return 1/IG(asiggamma, 1/bsiggamma) # inverse gamma RVs
    
    
    def weights(self, MU, BSIG, sigma, index):
        # Sample theta from Normal distribution:
        beta = np.zeros(self.nl)
        thetar = mvrv(MU, np.dot(sigma, BSIG))                                    
        beta[index] = thetar
        print(f"theta = {beta}\n\n")
        return beta, thetar
    
    
    def latent_variable(self, D, y, zz, vs, p0):
        # sample z from the Bernoulli distribution:
        lat = zz.copy()
        for jj in range(self.nl):
            ztemp_0 = zz
            ztemp_0[jj] = 0
            if np.mean(ztemp_0) == 0:
                PZ0 = self.pyzv0(y)
            else:
                PZ0 = self.pyzv(D, ztemp_0, vs, y)
            
            ztemp_1 = zz
            ztemp_1[jj] = 1      
            PZ1 = self.pyzv(D, ztemp_1, vs, y)
            
            zeta = PZ0 - PZ1  
            zeta = p0/( p0 + np.exp(zeta)*(1-p0))
            zz[jj] = bern(1, p = zeta, size = None)
        if np.all(zz == False):
            print("null sumple.")
            return lat
        return zz
    
    
    def augment(self, x, indices):
        x0 = x[0,:]
        x_total = x.copy()
        for tau in indices:
            # print('type of tau = ', type(tau))
            trash = np.zeros(x.shape)
            for i in range(len(x)):
                if i < tau:
                    trash[i,:] = x0
                else:
                    trash[i,:] = x[i-tau,:]
            x_total = np.concatenate([x_total, trash], axis=1)
        return x_total

    
    def pyzv_g(self, ztemp, vs, y, mean, covar, covar_inv):
        Sz = sum(ztemp)
        BSIG = covar
        BSIG_inv = covar_inv
        (sign, logdet1) = LA.slogdet(BSIG)                                              
        
        PZ = LG(self.asig + 0.5*self.N) -0.5*self.N*np.log(2*np.pi) - 0.5*Sz*np.log(vs) \
                + self.asig*np.log(self.bsig) - LG(self.asig) + 0.5*logdet1
        denom =  0.5*(np.matmul(y.T,y) - np.matmul(np.matmul(mean.T, BSIG_inv), mean))
        
        PZ = PZ - (self.asig+0.5*self.N)*(np.log(self.bsig + denom))
        
        return PZ   
    
    
    def update_library(self, x, latent, y , vs, taus):
        new_taus = np.zeros(self.num_delays, int)
        for index, nalpha in enumerate(self.Nalphas):
            # print('tau number ', index)
            print('nalpha = ', nalpha)
            print('start index = ', self.start_indices[index])
            
            counter = np.zeros(nalpha)
            
            trash = self.G[index].copy().ravel()
            divisors = np.zeros(nalpha)
            mult = np.zeros(nalpha).ravel()
            
            if trash.size != nalpha:
                print('trash size = ', trash.size)
                raise ValueError('trash and nalpha mismatch.')
            
            for i in range(nalpha):
                tau_temps = taus.copy()
                tau_temps[index] = self.start_indices[index]+i
                temp_aug = self.augment(x, tau_temps)
                # print('temp aug shape = ', temp_aug.shape)
                temp_library = self.transform(temp_aug)
                temp_mean, temp_covar, _,_, temp_covar_inv = self.Bsig_and_Bmu(latent, temp_library, vs, y)
                divisors[i] = self.pyzv_g(latent, vs, y, mean=temp_mean, covar=temp_covar, covar_inv=temp_covar_inv)
            
            for j in range(nalpha):
                check = 1/(np.sum(np.exp(divisors + np.log(trash) - divisors[j])))
                mult[j] = check
                if np.isnan(check):
                    raise ValueError('nan in mult.')
                if check == 0:
                    counter[j] = 1
            
            ZETA = mult*trash
            
            counter[np.where(ZETA<self.G_threshold)[0]] = 1
                
            if np.any(ZETA<0):
                raise ValueError('ZETA contains a value less than 0.')
            
            if np.any(ZETA > 1):
                ZETA[np.where(ZETA>1)[0]] = 1
                
            if np.any(ZETA==1):
                print(f'Tau{index} converged.')
                counter[np.where(ZETA!=1)] = 1
                
            if np.any(np.isnan(ZETA)):
                raise ValueError('ZETA contains NaNs.')
            
            max = np.where(ZETA==ZETA.max())[0][0]
            if np.any(counter==1):
                for i in range(max, nalpha):
                    if counter[i] == 1:
                        counter[i:] = 1
                        break
                # print('counter=',counter)
                lower = np.where(counter==0)[0].min()
                upper = np.where(counter==0)[0].max()
                print(f'Tau{index}: lower = {lower}')
                print(f'Tau{index}: upper = {upper}')
                print('counter sum = ', counter.sum())
                
                self.start_indices[index] += lower
                
                self.G[index] = self.G[index].ravel()[lower:upper+1]
                
                self.G_prior[index] = self.G_prior[index].ravel()[lower:upper+1]
                
                # print('updated self.G_prior shape = ', self.G_prior[index].shape)
                # print('updated self.G shape = ', self.G[index].shape)
                
                self.Nalphas[index] = upper + 1 - lower
                
                print(f'Tau{index}: new Nalpha = {self.Nalphas[index]}')
                zeta_new = ZETA[lower:upper+1]
                zeta_max = zeta_new.max()
                zeta_max_index = np.where(zeta_new==zeta_max)[0]
            else:
                zeta_new = ZETA.copy()
                zeta_max = zeta_new.max()
                zeta_max_index = np.where(zeta_new==zeta_max)[0]
                            
            zeta_new = self.normalize_g(zeta_new)
            
            update = multinomial.rvs(1,zeta_new)
            sampled_tau = np.where(update==1)[0][0].item()
            indd = sampled_tau + self.start_indices[index]
            print('indd = ', indd)
            print('sampled tau = ', sampled_tau, '\n')
            new_taus[index] = indd
        
        updated_augment = self.augment(x, new_taus)
        updated_library = self.transform(updated_augment)
        print('Library updated.\n')
        return updated_library, new_taus
    
    
    def dirichlet(self,indices):
        # indices = self.tau - self.start
        updated_G = []
        for priors in range(self.num_delays):
            # print(f'Index being updated in Tau{priors}: {indices[priors]}')
            trash = self.G_prior[priors].copy()
            trash[indices[priors]] += 1
            updated_alphas = dirichlet.rvs(trash)
            updated_G.append(updated_alphas)
        return updated_G
    
    
    def transform(self, data):
        return np.array(self.Library.transform(data))
    
    
    def normalize_g(self, G):
        if np.any(G>1):
            print(np.where(G>1))
            raise ValueError('ZETA value more than 1.')
        
        if np.any(G<0):
            print(np.where(G<0))
            raise ValueError('ZETA value less than 0.')
        
        if np.any(G == 0):
            G[np.where(G==0)] = 1e-300
        
        sum = G.sum()
        g = np.array(G)
        # print('type inside normalize = ', type(g))
        if sum > 1:
            balance = sum - 1
            # print('balance = ', balance)
            g[np.where(g==np.max(g))] = g.max() - balance #- self.G_threshold
            # print('new sum = ', g.sum())
            return np.array(g)
        else:
            return g
            
    
    
    def forward(self, X, Y, verbose=False, verbose_interval=50):
        # print('self.tau shape = ', self.taus.shape)
        if not self.known_tau:
            self.taus[:,0] = (np.arange(self.num_delays)+1)*500
            # print('thing size = ', self.taus[:,0].size)
            aug = self.augment(X, self.taus[:,0])
            Dict = self.transform(aug)
            print('design matrix shape =', Dict)
        else:
            Dict = X.copy()
        
        self.sigma[0] = self.res_var(Dict, Y)
        zint = self.linreg(Dict, Y)
        zval = zint.copy()
        print('zval shape = ', zval.shape)
        Bmu, BSIG, Aor, zr, _ = self.Bsig_and_Bmu(zval, Dict, self.v[0], Y)
        _, thetar = self.weights(Bmu, BSIG, self.sigma[0], zr)
        
        print('dict shape = ', Dict.shape)
        print('dict correlation = ', np.corrcoef(Dict, rowvar=False).shape)
        
        for epoch in range(1, self.MCMC):
            if verbose:
                if epoch % verbose == 0:
                    print('Iteration - {}'.format(epoch))
                    
            if not self.known_tau:
                d, indexx = self.update_library(x=X, latent=zval, y=Y,
                                                vs=self.v[epoch-1], taus=self.taus[:,epoch-1])
                D = d.copy()
                if epoch > self.burn_in:
                    indexx = np.round(np.mean((self.taus[:,int(epoch/2):epoch]),axis=1))
                    self.known_tau = True
                    print('tau converged.')
            # print('D shape = ', D.shape)
            
            self.taus[:,epoch] = indexx
            taus = self.taus[:,epoch] * self.dt
            print('taus = ', taus)
            print('tau inds = ', indexx)
            
            zr = zval.copy()
            zr = self.latent_variable(D, Y, zr, self.v[epoch-1], self.p0[epoch-1])
            zval = zr.copy()
            
            if epoch >= self.burn_in:
                self.zstore[:, epoch-self.burn_in] = zval       
            
            self.sigma[epoch] = self.noise_sig(Bmu, BSIG, Y)
            
            Sz = sum(zval)
            self.v[epoch] = self.slab(thetar, Aor, Sz, self.sigma[epoch])
            
            self.p0[epoch] = self.binary(Sz)
            
            if not self.known_tau:
                self.G = self.dirichlet(self.taus[:,epoch]-self.start_indices)
            
            Bmu, BSIG, Aor, index, _ = self.Bsig_and_Bmu(zval, D, self.v[epoch], Y)
            if len(Bmu) == 0:
                raise ValueError("bmu empty")
            weights, thetar = self.weights(Bmu, BSIG, self.sigma[epoch], index)
            if epoch > self.burn_in-1:
                self.theta[:, epoch-self.burn_in] = weights
            
            print('********************************************************\n')
                
        mean_theta = np.mean(self.theta, axis=-1)
        mean_latent = np.mean(self.zstore, axis=-1)
        mean_tau = self.taus[:,self.burn_in:]
        function_names = np.array(self.Library.get_feature_names())
        sig_theta = np.var(self.theta, axis=1)
        return mean_theta, mean_latent, sig_theta, mean_tau, function_names, self.theta, self.taus