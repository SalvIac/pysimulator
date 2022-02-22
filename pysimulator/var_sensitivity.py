# -*- coding: utf-8 -*-
# pysimulator
# Copyright (C) 2021-2022 Salvatore Iacoletti
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
"""

import numpy as np
from tqdm import tqdm 
from SALib.test_functions import Ishigami
from pysimulator.run_multiprocess import run_multiprocess


# def fun(*args):
#     return args[0]*args[1]**2 - args[2]**2

# Sobol G function (Saltelli et al. 2010)
class SololG():
    def __init__(self, *args):
        self.a_coeffs = np.array(args)
    def __call__(self, *x):
        x = np.array(x)
        return np.prod(self.gi(x))
    def gi(self, x):
        return (np.abs(4*x-2)+self.a_coeffs)/(1+self.a_coeffs)

# # Shifted Sobol G function (Saltelli et al. 2010)
# class ShiftSololG():
#     def __init__(self, *args):
#         self.a_coeffs = np.array(args)
#     def __call__(self, x):
#         x = np.array(x)
#         return np.prod(self.gi(x))
#     def gi(self, x):
#         return (np.abs(4*x-2)+self.a_coeffs)/(1+self.a_coeffs)
        
def ishigami_interface(*args):
    return Ishigami.evaluate(np.array(args).reshape((1,len(args))))

def ishigami_interface2(args):
    return ishigami_interface(*args)



class VarBasedSensitivity():
    
    multiprocessing = False
    cores = 4
    
    def __init__(self, A, B, dependencies=None):
        # check
        if A.shape[0] != B.shape[0]:
            raise Exception("the number of simulation cannot be different between A and B matrix")
        if A.shape[1] != B.shape[1]:
            raise Exception("the number of input variables cannot be different between A and B matrix")
        self.A = A
        self.B = B
        self.N = A.shape[0]
        self.num_inputs = A.shape[1]
        # in case some of the inputs (columns) are dependent on each other
        # example {1:[2,3]} # columns 2 and 3 are dependent on 1
        self.dependencies = dependencies
    
    
    def precompute(self, fun):
        print("precompute ya and yb")        
        if self.multiprocessing:
            inputs_A = list()
            inputs_B = list()
            for row in range(0, self.N):
                inputs_A.append([*self.A[row,:]])
                inputs_B.append([*self.B[row,:]])
            ya_pre = run_multiprocess(fun, inputs_A, self.cores)
            yb_pre = run_multiprocess(fun, inputs_B, self.cores)
        else:
            ya_pre, yb_pre = list(), list()
            for row in tqdm(range(0, self.N)):
                ya_pre.append( fun(*self.A[row,:]) )
                yb_pre.append( fun(*self.B[row,:]) )
        print("precompute yci and ydi")
        yci_pre, ydi_pre = dict(), dict()
        for ith in range(0, self.num_inputs):
            Ci, Di = self.get_c_d_matrix(ith, self.A, self.B)
            if self.multiprocessing:
                inputs_Ci = list()
                inputs_Di = list()
                for row in range(0, self.N):
                    inputs_Ci.append([*Ci[row,:]])
                    inputs_Di.append([*Di[row,:]])
                yci_pre[ith] = run_multiprocess(fun, inputs_Ci, self.cores)
                ydi_pre[ith] = run_multiprocess(fun, inputs_Di, self.cores)
            else:
                yci_pre[ith] = list()
                ydi_pre[ith] = list()
                for row in tqdm(range(0, self.N)):
                    yci_pre[ith].append( fun(*Ci[row,:]) )
                    ydi_pre[ith].append( fun(*Di[row,:]) )
        self.ya_pre = ya_pre
        self.yb_pre = yb_pre
        self.yci_pre = yci_pre
        self.ydi_pre = ydi_pre

    
    def run(self, discriminate=None, output=2):
        if discriminate is None:
            discriminate = self._discriminate
        results = list()
        for ith in range(0, self.num_inputs):
            ya = discriminate(self.ya_pre)
            yb = discriminate(self.yb_pre)
            yci = discriminate(self.yci_pre[ith])
            ydi = discriminate(self.ydi_pre[ith])
            results.append( self.sampling_sensitivity(ya, yb, yci, ydi)[output] )
        return np.array(results)
    
    
    # this is to check
    def get_y(self, fun, ith):
        Ci, Di = self.get_c_d_matrix(ith, self.A, self.B)
        ya, yb, yci, ydi = list(), list(), list(), list()
        for row in range(0, self.N):
            ya.append( fun(*self.A[row,:]) )
            yb.append( fun(*self.B[row,:]) )
            yci.append( fun(*Ci[row,:]) )
            ydi.append( fun(*Di[row,:]) )
        return np.array(ya), np.array(yb), np.array(yci), np.array(ydi)
        
    
    def sampling_sensitivity(self, ya, yb, ydi, yci):
        sum_AC = np.sum( ya*yci )
        sum_BC = np.sum( yb*yci )
        sum_BD_1 = np.sum( (yb - ydi)**2 )
        sum_BD_2 = np.sum( yb**2 + ydi**2 )
        sum_BD_3 = np.sum( yb*ydi )
        sum_AD_3 = np.sum( ya*ydi )
        sum_AB = np.sum( ya**2 + yb**2 )
        sum_AB_2 = np.sum( ya + yb )
        sum_ABCD = np.sum( ya*yb + yci*ydi )        
        
        fo = np.sum(ya)/self.N
        fo2 = (1/(2*self.N)) * sum_ABCD
        si1 = (sum_AC/self.N - fo**2) / ((np.sum(ya**2)/self.N) - fo**2)
        si2 = 1 - ((1/self.N)*sum_BD_1) / (((1/self.N)*sum_BD_2) - (np.mean(yb)**2)-(np.mean(ydi)**2))
        si3 = (((1/(2*self.N))*(sum_AC + sum_BD_3)) - fo2) / (((1/(2*self.N))*sum_AB) - fo2)
        sti3 = 1 - ((((1/(2*self.N))*(sum_BC + sum_AD_3)) - fo2) / (((1/(2*self.N))*sum_AB) - fo2))
        e3 = sti3*(((1/(2*self.N))*sum_AB) - fo2)
        return si1, si2, si3, sti3, e3
        
    
    def get_c_d_matrix(self, ith, A, B):
        Ci = self.substitute_ith_col(ith, A.copy(), B.copy())
        Di = self.substitute_ith_col(ith, B.copy(), A.copy())
        return Ci, Di
    
    
    def substitute_ith_col(self, ith, m1, m2):
        m1[:,ith] = m2[:,ith]
        # if ith in dependencies:
        #     for i in dependencies[ith]:
        #         m1[:,i] = m2[:,i]
        return m1    
    
    
    # this has to always output a numpy array
    @classmethod
    def _discriminate(cls, a):
        return np.array(a) # this is the identity    






#%% testing from here https://github.com/SALib/SALib
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    '''
    In [2], the expected first-order indices are:
    x1: 0.3139
    x2: 0.4424
    x3: 0.0
    '''
    np.random.seed(1992)
    num_simulations = 100000
    
    A = -np.pi + (np.random.rand(num_simulations,3) * (np.pi - -np.pi)) 
    B = -np.pi + (np.random.rand(num_simulations,3) * (np.pi - -np.pi)) 

    # fig. 2 Saltelli et al. (2010)
    # x = np.array([0.,0.5,1.])
    # fig = plt.figure()
    # fun = SololG(*[0.,0.,0.])
    # plt.plot(x, fun.gi(x))
    # fun = SololG(*[1.,1.,1.])
    # plt.plot(x, fun.gi(x))
    # fun = SololG(*[9.,9.,9.])
    # plt.plot(x, fun.gi(x))
    # plt.show()
    
    
    vbs = VarBasedSensitivity(A, B)
    
    vbs.precompute(ishigami_interface)
    re = vbs.run()
    print(re)

    vbs.multiprocessing = True
    vbs.precompute(ishigami_interface2)
    re = vbs.run()
    print(re)


    # vbs.get_c_d_matrix(0, A, B)

    # substitute ith column of matrix A with the ith column matrix B
    



#%%

# function [Si_1,Si_2,Si_3,Sti_3,E_3] =  SamplingSensitivities(A,B,C,D,N)

# sum_AC=0
# sum_BD_1=0
# sum_BD_2=0
# sum_BD_3=0
# sum_AD_3=0
# sum_AB=0
# sum_BC=0

# sum_AB_2=0;
# sum_ABCD=0;
# for i=1:N
#     sum_AC=sum_AC+A(i)*C(i);
#     sum_BC=sum_BC+B(i)*C(i);
#     sum_BD_1 = sum_BD_1+(B(i)-D(i))^2;
#     sum_BD_2 = sum_BD_2+(B(i)^2+D(i)^2);
#     sum_BD_3 = sum_BD_3+B(i)*D(i);
#     sum_AD_3 = sum_AD_3+A(i)*D(i);
#     sum_AB = sum_AB+(A(i)^2+B(i)^2);
#     sum_AB_2 = sum_AB_2+ A(i)+B(i);
#     sum_ABCD = sum_ABCD+(A(i)*B(i))+(C(i)*D(i));
# end
# fo = sum(A)/N;
# fo2 = (1/(2*N))*(sum_ABCD);

# Si_1 = (((1/N)*sum_AC) -fo^2)/((sum(A.^2)/N)-fo^2);

# Si_2 = 1-((1/N)*sum_BD_1)/(((1/N)*sum_BD_2)-(mean(B)^2)-(mean(D)^2));
# Si_3 = (((1/(2*N))*(sum_AC+sum_BD_3))-fo2)/(((1/(2*N))*sum_AB)-fo2);
# Sti_3 =1- ((((1/(2*N))*(sum_BC+sum_AD_3))-fo2)/(((1/(2*N))*sum_AB)-fo2));

# E_3 = Sti_3*((((1/(2*N))*sum_AB)-fo2));

