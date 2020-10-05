# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 19:06:13 2018

@author: Akshay Ghosh
"""

#######################################################################
#EXTENSIONS

#import pandas as pd
#import lxml
#import pickle
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
from matplotlib import style
import math
style.use('fivethirtyeight')
import scipy as sp
from scipy import stats

#######################################################################
#FUNCTIONS

def npa(X):
    return np.array(X)

def fig_num():
    fig_num = np.random.randint(1,100000)
    return fig_num

def get_unification(mass_param):
    
    P = standard_model(mass_gen_4 = mass_param[0],mass_gen_5 = mass_param[1],mass_gen_6 = mass_param[2],mass_gen_7 = mass_param[3])
    u_quality = plot_coupling(P,plot = True)
    #print(u_quality)
    
    return u_quality

def plot_coupling(P, plot = True):
    
    n = 10000
    #X = np.linspace(91.1876,pow(10,48), n) # try to make this non linear
    X = np.geomspace(91.1876,pow(10,48), n) # THIS WAS THE PROBLEM (when it was linear)
    #global X_sqr
    X_sqr = pow(X,0.5)

    
    # here group the particles by mass
    # Y_i are now going to be multidimensional lists
    P_g = group_by_mass(P,X) # P_g is same length as X
    P_g_l = len(P_g)
    
    Y_1 = np.zeros(P_g_l); Y_2 = np.zeros(P_g_l); Y_3 = np.zeros(P_g_l)
    
    for i in range(P_g_l):
        Y_1[i] = running_coupling_constants(X[i], P_g[i], k = 1)
        Y_2[i] = running_coupling_constants(X[i], P_g[i], k = 2)
        Y_3[i] = running_coupling_constants(X[i], P_g[i], k = 3)
    
    Y_1 = remove_discontinuity(Y_1)
    Y_2 = remove_discontinuity(Y_2)
    Y_3 = remove_discontinuity(Y_3)
    
    u_quality = unification_quality(X_sqr,Y_1,Y_2,Y_3)
    s = ('unification quality: {} \n(perfect is 0, SM is 1)'.format(u_quality))
    
    #plot_b_value(X,P)
    if plot == True:
        plt.figure(fig_num())
        plt.plot(X_sqr,Y_1,color = 'red',label = r'$1/\alpha_1$, U(1)')
        plt.plot(X_sqr,Y_2,color = 'blue',label = r'$1/\alpha_2$, SU(2)')
        plt.plot(X_sqr,Y_3,color = 'green',label = r'$1/\alpha_3$, SU(3)')
        plt.xscale('log',basex = 10)
        plt.ylim(8,60)
        plt.title('Runnning Coupling Constants {}'.format(title_str))
        plt.xlabel(r'$Q$ (GeV)')
        plt.ylabel(r'$ 1/\alpha_i}$')
        plt.text(5,20,s,bbox=dict(facecolor='yellow', alpha=0.5))
        plt.legend()
        plt.show()
    
    return u_quality

def running_coupling_constants(Q,P,k):
    '''
    Q is the independent variable (Q)
    P is particle content
    '''
    
    M_z = 91.1876 # mass of Z boson in Gev
    pi = math.pi
    
    b = (1/(16*pi))*npa([b_value(P,'U(1)'),b_value(P,'SU(2)'),b_value(P,'SU(3)')])
    c = npa([59.0,30.0,8.5])
    
    k -= 1 # for alpha_1,2,3
    
    f = c[k] - 4*b[k]*math.log10(pow((Q/M_z),2))
        
    return f

def b_value(X,G):
    '''
    X = particle content
    G = group (U(1), SU(2) or SU(3) )
    
    C_2 = quadratic casimir
    n_s = # scalars
    n_s = # fermions
    T_rs = casimir invariant for group G under rep r
    '''
    
    T_r = 0.5
    if G == "U(1)":
        b = count_charge_u1(X)
        
    elif G == "SU(2)" or G == "SU(3)":
        if G == "SU(2)":
            weyl_or_dirac = 2.0
            n_s = 1
        elif G == "SU(3)":
            weyl_or_dirac = 4.0
            #weyl_or_dirac = 2.0
            n_s = 0
        
        n_f = count_fermions(X,G)
        b = -((11.0/3)*quadratic_casimir(G) - (1.0/3)*n_s*T_r - (weyl_or_dirac/3)*n_f*T_r)
    
    return b

def count_fermions(X,G):
    if G == "SU(2)":
        n_f = count_fermions_su2(X)
    elif G == "SU(3)":
        n_f = count_fermions_su3(X)
    return(n_f)

def count_charge_u1(X):
    '''
    b_1 = (2/3)*[sum over fermions, weakhypercharge^2] + 
        (1/3)*[sum over scalars, weakhypercharge^2]
    '''
    # ^^ remember that the last index of X is scalars
    
    q_f_sqr = 0 # fermion weak hypercharge squared to be counted

    q_s_sqr = 2*pow(X[len(X)-1].weak_hypercharge,2)
                    
    for i in range(len(X) - 1):
        q_f_sqr += pow(X[i].weak_hypercharge,2)
    
    b_1 = (3.0/5)*((2.0/3)*q_f_sqr + (1.0/3)*q_s_sqr) # 3/5 is some normalization
    return(b_1)

def count_fermions_su2(X):
    '''
    sum over i is to sum over each generation. sum over j is to add the quarks,
    then the leptons. the second index (top/bottom type) is 0 because since
    under SU(2) you can transform eg u -> d, they count as the "same particle"
    '''
    n_f_su2 = 0    
    A = [] # will fill this list then count it for # fermions under SU(2)
    
    for i in range(len(X)):
        if X[i].chirality == -1:
            A.append(X[i]) # list of all LH particles
    
    double_count_check = [] # fill this list with the [i,j] where its been counted
    
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i].colour == A[j].colour:
                if A[i].su2_isospin == -A[j].su2_isospin:
                    if A[i].weak_hypercharge == A[j].weak_hypercharge:
                        if A[i].mass == A[j].mass:
                            if [i,j] not in double_count_check:
                                #s_p(A[i])
                                #s_p(A[j])
                                #print('')
                                n_f_su2 += 1
                                double_count_check.append([j,i])                               
    
    return(n_f_su2)

def count_fermions_su3(X):
    # count SU(3) fermions:
    n_f_su3 = 0
    
    for i in range(len(X)):
        if X[i].colour == 'r' and X[i+1].colour == 'b' and X[i+2].colour == 'g':
            if X[i].chirality == -1 and X[i+1].chirality == -1 and X[i+2].chirality == -1:
                n_f_su3 += 1
    
    return(n_f_su3)

def quadratic_casimir(G):
    '''
    given a group SU(N) return the quadratic casimir N
    '''
    if G == "U(1)":
        C_2 = 0
    elif G == "SU(2)":
        C_2 = 2
    elif G == "SU(3)":
        C_2 = 3
    
    return(C_2)

def group_by_mass(X,E):
    '''
    given particles X (X is a list, not a np vector), group them by 2M < E
    ie if 2M > E: exclude
    
    Q is energy
    '''
    E_l = len(E)
    X_l = len(X)
    
    X_g = [] # particles grouped by mass
    X_g_temp = []
    
    for i in range(E_l):
        for j in range(X_l):
            if (2*X[j].mass)**2 < E[i]: # include
                X_g_temp.append(X[j])
        X_g.append(X_g_temp)
        X_g_temp = []
    
    return(X_g)


def standard_model(mass_gen_4 = 7e3,mass_gen_5 = 1e4,mass_gen_6 = 1e5, mass_gen_7 = 3000):
    '''
    how to interpret for some X which is all the particles:
    X[generation #] (for now scalars are generation 4)
     [SU(2) selection ie "top" or "bottom" type]
     [quark or lepton]
     [SU(3) selection ie colour]
    '''    
    # particle = (colour,SU(2) isospin, weak hypercharge, chirality, mass)
    mass_gen_1 = 0.01
    mass_gen_2 = 0.02
    mass_gen_3 = 0.03
#    mass_gen_4 = 1e3
#    mass_gen_5 = 1e4
#    mass_gen_6 = 1e5
    #mass_gen_4 += 0.04
    mass_gen_4 = 0.04
    mass_gen_5 += 0.05
    #mass_gen_5 = 0.05
    mass_gen_6 += 0.06
    mass_gen_7 += 0.07
    mass_gen_8 = 1280 + 0.05
    
    # scalars:
    h = Particle(None,2.0,0.5,1,0)
    
    # left handed fermions:
    u_r = Particle('r',0.5,1.0/6,-1, mass_gen_1) # GEN 1 L
    u_b = Particle('b',0.5,1.0/6,-1, mass_gen_1)
    u_g = Particle('g',0.5,1.0/6,-1, mass_gen_1)
    e = Particle(None,-0.5,-1.0/2,-1, mass_gen_1)
    
    d_r = Particle('r',-0.5,1.0/6,-1, mass_gen_1)
    d_b = Particle('b',-0.5,1.0/6,-1, mass_gen_1)
    d_g = Particle('g',-0.5,1.0/6,-1, mass_gen_1)
    ve = Particle(None,0.5,-1.0/2,-1, mass_gen_1)
    
    
    c_r = Particle('r',0.5,1.0/6,-1, mass_gen_2) # GEN 2 L
    c_b = Particle('b',0.5,1.0/6,-1, mass_gen_2)
    c_g = Particle('g',0.5,1.0/6,-1, mass_gen_2)
    mu = Particle(None,-0.5,-1.0/2,-1, mass_gen_2)
    
    s_r = Particle('r',-0.5,1.0/6,-1, mass_gen_2)
    s_b = Particle('b',-0.5,1.0/6,-1, mass_gen_2)
    s_g = Particle('g',-0.5,1.0/6,-1, mass_gen_2)
    vmu = Particle(None,0.5,-1.0/2,-1, mass_gen_2)
    

    t_r = Particle('r',0.5,1.0/6,-1, mass_gen_3) # GEN 3 L
    t_b = Particle('b',0.5,1.0/6,-1, mass_gen_3)
    t_g = Particle('g',0.5,1.0/6,-1, mass_gen_3)
    tau = Particle(None,-0.5,-1.0/2,-1, mass_gen_3)
    
    b_r = Particle('r',-0.5,1.0/6,-1, mass_gen_3)
    b_b = Particle('b',-0.5,1.0/6,-1, mass_gen_3)
    b_g = Particle('g',-0.5,1.0/6,-1, mass_gen_3)
    vtau = Particle(None,0.5,-1.0/2,-1, mass_gen_3)

    # right handed fermions:
    u_r_r = Particle('r',0.5,2.0/3,1,mass_gen_1) # GEN 1 R
    u_b_r = Particle('b',0.5,2.0/3,1,mass_gen_1)
    u_g_r = Particle('g',0.5,2.0/3,1,mass_gen_1)
    e_r = Particle(None,-0.5,-1.0,1,mass_gen_1)
    
    d_r_r = Particle('r',-0.5,-1.0/3,1,mass_gen_1)
    d_b_r = Particle('b',-0.5,-1.0/3,1,mass_gen_1)
    d_g_r = Particle('g',-0.5,-1.0/3,1,mass_gen_1)
    ve_r= Particle(None,0.5,0,1,mass_gen_1)
    # note theres no right handed neutrinos in the SM
    
    c_r_r = Particle('r',0.5,2.0/3,1,mass_gen_2) # GEN 2 R
    c_b_r = Particle('b',0.5,2.0/3,1,mass_gen_2)
    c_g_r = Particle('g',0.5,2.0/3,1,mass_gen_2)
    mu_r = Particle(None,-0.5,-1.0,1,mass_gen_2)
    
    s_r_r = Particle('r',-0.5,-1.0/3,1,mass_gen_2)
    s_b_r = Particle('b',-0.5,-1.0/3,1,mass_gen_2)
    s_g_r = Particle('g',-0.5,-1.0/3,1,mass_gen_2)
    vmu_r = Particle(None,0.5,0,1,mass_gen_2)
    
    
    t_r_r = Particle('r',0.5,2.0/3,1,mass_gen_3) # GEN 3 R
    t_b_r = Particle('b',0.5,2.0/3,1,mass_gen_3)
    t_g_r = Particle('g',0.5,2.0/3,1,mass_gen_3)
    tau_r = Particle(None,-0.5,-1.0,1,mass_gen_3)
    
    b_r_r = Particle('r',-0.5,-1.0/3,1,mass_gen_3)
    b_b_r = Particle('b',-0.5,-1.0/3,1,mass_gen_3)
    b_g_r = Particle('g',-0.5,-1.0/3,1,mass_gen_3)
    vtau_r = Particle(None,0.5,0,1,mass_gen_3)
    
    ########################################################################
    # now add in gen 4 L, gen 4 R
    t_r_4 = Particle('r',0.5,1.0/6,-1,mass_gen_4) # GEN 4 L
    t_b_4 = Particle('b',0.5,1.0/6,-1,mass_gen_4)
    t_g_4 = Particle('g',0.5,1.0/6,-1,mass_gen_4)
    tau_4 = Particle(None,-0.5,-1.0/2,-1,mass_gen_4)
    
    b_r_4 = Particle('r',-0.5,1.0/6,-1,mass_gen_4)
    b_b_4 = Particle('b',-0.5,1.0/6,-1,mass_gen_4)
    b_g_4 = Particle('g',-0.5,1.0/6,-1,mass_gen_4)
    vtau_4 = Particle(None,0.5,-1.0/2,-1,mass_gen_4)
    
    t_r_r_4 = Particle('r',0.5,2.0/3,1,mass_gen_4) # GEN 4 R
    t_b_r_4 = Particle('b',0.5,2.0/3,1,mass_gen_4)
    t_g_r_4 = Particle('g',0.5,2.0/3,1,mass_gen_4)
    tau_r_4 = Particle(None,-0.5,-1.0,1,mass_gen_4)
    
    b_r_r_4 = Particle('r',-0.5,-1.0/3,1,mass_gen_4)
    b_b_r_4 = Particle('b',-0.5,-1.0/3,1,mass_gen_4)
    b_g_r_4 = Particle('g',-0.5,-1.0/3,1,mass_gen_4)
    vtau_r_4 = Particle(None,0.5,0,1,mass_gen_4)
    
    ########################################################################
    # now add in gen 5 L, gen 5 R
    t_r_5 = Particle('r',0.5,1.0/6,-1,mass_gen_5) # GEN 5 L
    t_b_5 = Particle('b',0.5,1.0/6,-1,mass_gen_5)
    t_g_5 = Particle('g',0.5,1.0/6,-1,mass_gen_5)
    tau_5 = Particle(None,-0.5,-1.0/2,-1,mass_gen_5)
    
    b_r_5 = Particle('r',-0.5,1.0/6,-1,mass_gen_5)
    b_b_5 = Particle('b',-0.5,1.0/6,-1,mass_gen_5)
    b_g_5 = Particle('g',-0.5,1.0/6,-1,mass_gen_5)
    vtau_5 = Particle(None,0.5,-1.0/2,-1,mass_gen_5)
    
    t_r_r_5 = Particle('r',0.5,2.0/3,1,mass_gen_5) # GEN 5 R
    t_b_r_5 = Particle('b',0.5,2.0/3,1,mass_gen_5)
    t_g_r_5 = Particle('g',0.5,2.0/3,1,mass_gen_5)
    tau_r_5 = Particle(None,-0.5,-1.0,1,mass_gen_5)
    
    b_r_r_5 = Particle('r',-0.5,-1.0/3,1,mass_gen_5)
    b_b_r_5 = Particle('b',-0.5,-1.0/3,1,mass_gen_5)
    b_g_r_5 = Particle('g',-0.5,-1.0/3,1,mass_gen_5)
    vtau_r_5 = Particle(None,0.5,0,1,mass_gen_5)
    
    ########################################################################
    # now add in gen 6 L, gen 6 R
    t_r_6 = Particle('r',0.6,1.0/6,-1,mass_gen_6) # GEN 6 L
    t_b_6 = Particle('b',0.6,1.0/6,-1,mass_gen_6)
    t_g_6 = Particle('g',0.6,1.0/6,-1,mass_gen_6)
    tau_6 = Particle(None,-0.6,-1.0/2,-1,mass_gen_6)
    
    b_r_6 = Particle('r',-0.6,1.0/6,-1,mass_gen_6)
    b_b_6 = Particle('b',-0.6,1.0/6,-1,mass_gen_6)
    b_g_6 = Particle('g',-0.6,1.0/6,-1,mass_gen_6)
    vtau_6 = Particle(None,0.6,-1.0/2,-1,mass_gen_6)
    
    t_r_r_6 = Particle('r',0.6,2.0/3,1,mass_gen_6) # GEN 6 R
    t_b_r_6 = Particle('b',0.6,2.0/3,1,mass_gen_6)
    t_g_r_6 = Particle('g',0.6,2.0/3,1,mass_gen_6)
    tau_r_6 = Particle(None,-0.6,-1.0,1,mass_gen_6)
    
    b_r_r_6 = Particle('r',-0.6,-1.0/3,1,mass_gen_6)
    b_b_r_6 = Particle('b',-0.6,-1.0/3,1,mass_gen_6)
    b_g_r_6 = Particle('g',-0.6,-1.0/3,1,mass_gen_6)
    vtau_r_6 = Particle(None,0.6,0,1,mass_gen_6)
    
    ########################################################################
    # now add in gen 7 L, gen 7 R
    t_r_7 = Particle('r',0.7,1.0/7,-1,mass_gen_7) # GEN 7 L
    t_b_7 = Particle('b',0.7,1.0/7,-1,mass_gen_7)
    t_g_7 = Particle('g',0.7,1.0/7,-1,mass_gen_7)
    tau_7 = Particle(None,-0.7,-1.0/2,-1,mass_gen_7)
    
    b_r_7 = Particle('r',-0.7,1.0/7,-1,mass_gen_7)
    b_b_7 = Particle('b',-0.7,1.0/7,-1,mass_gen_7)
    b_g_7 = Particle('g',-0.7,1.0/7,-1,mass_gen_7)
    vtau_7 = Particle(None,0.7,-1.0/2,-1,mass_gen_7)
    
    t_r_r_7 = Particle('r',0.7,2.0/3,1,mass_gen_7) # GEN 7 R
    t_b_r_7 = Particle('b',0.7,2.0/3,1,mass_gen_7)
    t_g_r_7 = Particle('g',0.7,2.0/3,1,mass_gen_7)
    tau_r_7 = Particle(None,-0.7,-1.0,1,mass_gen_7)
    
    b_r_r_7 = Particle('r',-0.7,-1.0/3,1,mass_gen_7)
    b_b_r_7 = Particle('b',-0.7,-1.0/3,1,mass_gen_7)
    b_g_r_7 = Particle('g',-0.7,-1.0/3,1,mass_gen_7)
    vtau_r_7 = Particle(None,0.7,0,1,mass_gen_7)
    
    ########################################################################
    
    # scalar generations:
    s_1 = [[[h]]]
    
    # fermion generations of quark,lepton:
    f_1_l = [[[u_r,u_b,u_g],[e]],[[d_r,d_b,d_g],[ve]]] # LH
    f_2_l = [[[c_r,c_b,c_g],[mu]],[[s_r,s_b,s_g],[vmu]]]
    f_3_l = [[[t_r,t_b,t_g],[tau]],[[b_r,b_b,b_g],[vtau]]]
    
    f_1_r = [[[u_r_r,u_b_r,u_g_r],[e_r]],[[d_r_r,d_b_r,d_g_r],[]]] # RH
    f_2_r = [[[c_r_r,c_b_r,c_g_r],[mu_r]],[[s_r_r,s_b_r,s_g_r],[]]]
    f_3_r = [[[t_r_r,t_b_r,t_g_r],[tau_r]],[[b_r_r,b_b_r,b_g_r],[]]]

#    f_4_l = [[[t_r,t_b,t_g],[tau]],[[b_r,b_b,b_g],[vtau]]]
#    f_4_r = [[[t_r_r,t_b_r,t_g_r],[tau_r]],[[b_r_r,b_b_r,b_g_r],[]]]

#    f_4_l = [[[t_r_4,t_b_4,t_g_4],[tau_4]],[[b_r_4,b_b_4,b_g_4],[vtau_4]]]
#    f_4_r = [[[],[tau_r_4]],[[],[]]]
    
    f_5_l = [[[t_r_5,t_b_5,t_g_5],[tau_5]],[[b_r_5,b_b_5,b_g_5],[vtau_5]]]
    f_5_r = [[[],[tau_r_5]],[[],[]]]
    
    f_6_l = [[[t_r_6,t_b_6,t_g_6],[tau_6]],[[b_r_6,b_b_6,b_g_6],[vtau_6]]]
    f_6_r = [[[],[tau_r_6]],[[],[]]]
    
    f_7_l = [[[t_r_7,t_b_7,t_g_7],[tau_7]],[[b_r_7,b_b_7,b_g_7],[vtau_7]]]
    f_7_r = [[[],[tau_r_7]],[[],[]]]
#############
    f_4_l = [[[t_r_4,t_b_4,t_g_4],[tau_4]],[[b_r_4,b_b_4,b_g_4],[vtau_4]]]
    f_4_r = [[[t_r_r_4,t_b_r_4,t_g_r_4],[tau_r_4]],[[b_r_r_4,b_b_r_4,b_g_r_4],[]]]
    
#    f_4_l = [[[t_r_4,t_b_4,t_g_4],[]],[[b_r_4,b_b_4,b_g_4],[]]]
#    f_4_r = [[[t_r_r_4,t_b_r_4,t_g_r_4],[tau_r_4]],[[b_r_r_4,b_b_r_4,b_g_r_4],[]]]

#    f_4_l = [[[t_r_4,t_b_4,t_g_4],[tau_4]],[[b_r_4,b_b_4,b_g_4],[vtau_4]]]
#    f_4_r = [[[],[tau_r_4]],[[],[]]]
    
#    f_4_l = [[[t_r_4,t_b_4,t_g_4],[tau_4]],[[b_r_4,b_b_4,b_g_4],[vtau_4]]]
#    f_4_r = [[[t_r_r_4,t_b_r_4,t_g_r_4],[]],[[b_r_r_4,b_b_r_4,b_g_r_4],[]]]
    
#    f_4_l = [[[t_r_4,t_b_4,t_g_4],[tau_4]],[[b_r_4,b_b_4,b_g_4],[vtau_4]]]
#    f_4_r = [[[t_r_r_4,t_b_r_4,t_g_r_4],[tau_r_4]],[[b_r_r_4,b_b_r_4,b_g_r_4],[]]]
#    
#    f_5_l = [[[t_r_5,t_b_5,t_g_5],[tau_5]],[[b_r_5,b_b_5,b_g_5],[vtau_5]]]
#    f_5_r = [[[t_r_r_5,t_b_r_5,t_g_r_5],[tau_r_5]],[[b_r_r_5,b_b_r_5,b_g_r_5],[]]]
    
#    f_4_l = [[[],[tau_4]],[[],[vtau_4]]]
#    f_4_r = [[[t_r_r_4,t_b_r_4,t_g_r_4],[tau_r_4]],[[b_r_r_4,b_b_r_4,b_g_r_4],[]]]
#    
#    f_5_l = [[[],[tau_5]],[[],[vtau_5]]]
#    f_5_r = [[[t_r_r_5,t_b_r_5,t_g_r_5],[tau_r_5]],[[b_r_r_5,b_b_r_5,b_g_r_5],[]]]
#    
#    f_6_l = [[[],[tau_6]],[[],[vtau_6]]]
#    f_6_r = [[[t_r_r_6,t_b_r_6,t_g_r_6],[tau_r_6]],[[b_r_r_6,b_b_r_6,b_g_r_6],[]]]
#    
#    f_7_l = [[[],[tau_7]],[[],[vtau_7]]]
#    f_7_r = [[[t_r_r_7,t_b_r_7,t_g_r_7],[tau_r_7]],[[b_r_r_7,b_b_r_7,b_g_r_7],[]]]
    
#    f_6_l = [[[t_r_6,t_b_6,t_g_6],[tau_6]],[[b_r_6,b_b_6,b_g_6],[vtau_6]]]
#    f_6_r = [[[],[tau_r_6]],[[],[]]]
#    
#    f_7_l = [[[t_r_7,t_b_7,t_g_7],[tau_7]],[[b_r_7,b_b_7,b_g_7],[vtau_7]]]
#    f_7_r = [[[],[tau_r_7]],[[],[]]]
    
#    f_8_l = [[[t_r_8,t_b_8,t_g_8],[tau_8]],[[b_r_8,b_b_8,b_g_8],[vtau_8]]]
#    f_8_r = [[[],[tau_r_8]],[[],[vtau_r_8]]]
    
    #G = [f_1_l,f_2_l,f_3_l,s_1] # include only LH
    #G = [f_1_l,f_2_l,f_3_l,f_1_r,f_2_r,f_3_r,s_1] # 1st 3 gen
    G = [f_1_l,f_2_l,f_3_l,f_4_l,f_1_r,f_2_r,f_3_r,f_4_r,s_1] # 4th gen added
    #G = [f_1_l,f_2_l,f_3_l,f_4_l,f_1_r,f_2_r,f_3_r,f_4_r,f_5_l,f_5_r,s_1] # 5th gen added
    #G = [f_1_l,f_2_l,f_3_l,f_4_l,f_1_r,f_2_r,f_3_r,f_4_r,f_5_l,f_5_r,f_6_l,f_6_r,s_1] # 6th gen added
    
#    G = [f_1_l,f_2_l,f_3_l,f_4_l,f_1_r,f_2_r,f_3_r,f_4_r,f_5_l,\
#         f_5_r,f_6_l,f_6_r,f_7_l,f_7_r,s_1] # 7th gen added
 
    
    global title_str
    #title_str = '4th, 5th, 6th, 7th Generation Added (no RH quarks), m = 23000 GeV'
    title_str = '4th Generation Added'
    G_2 = []
    
    for i in range(len(G)):
        for j in range(len(G[i])): # this is f_1_l, etc
            for k in range(len(G[i][j])): # this is 'up' or 'down' type
                for l in range(len(G[i][j][k])): # this is quarks or lepton
                    G_2.append(G[i][j][k][l])
    
    
    return(G_2)

def show_particle(p):
    '''
    input a particle p, output its info from Particle class
    '''
    info = [p.colour,p.su2_isospin,p.weak_hypercharge,p.chirality,p.mass]
    return(info)
    
def s_p(p):
    print('(colour, su2 isospin, weak hypercharge, chirality, mass)')
    print(show_particle(p))
    return

def unification_quality(X,Y_1,Y_2,Y_3,Print = False):
    '''
    Y_i are the running for U(1), etc
    find the difference between Y_3(x), and where Y_1(x) = Y_2(x)
    '''
    
    n = len(X) # should be equal to len Y_i
    x_int_idx = 0
    threshold = 0.1
    # x intercept of Y_1,Y_2
    
    for i in range(n):
        if abs(Y_2[i] - Y_1[i]) < threshold:
            x_int_idx = i
            break
    
    Y_1_2_int = (Y_2[x_int_idx] + Y_1[x_int_idx])/2 # average so most accurate
    
    # THIS IS THE QUALITY:
    quality_SM = (6.20183486239)*(1.01572213269)
    absolute_quality = abs(Y_1_2_int - Y_3[x_int_idx])
    quality = absolute_quality/quality_SM
    
    if Print == True:
        print('STANDARD MODEL QUALITY: {}'.format(quality)) 
    
    return(quality)

def plot_b_value(X,P):
    '''
    
    '''
    #global P_g
    P_g = group_by_mass(P,X)
    #c = npa([59.0,30.0,8.5]) # initial c
    
    n = len(X)
    b1 = np.zeros(n); b2 = np.zeros(n); b3 = np.zeros(n)
    c1 = np.zeros(n); c2 = np.zeros(n); c3 = np.zeros(n)
    c1[0] = 59.0; c2[0] = 30.0; c3[0] = 8.5
    
    for i in range(n):
        b1[i] = b_value(P_g[i],'U(1)')
        b2[i] = b_value(P_g[i],'SU(2)')
        b3[i] = b_value(P_g[i],'SU(3)')

    
    plt.figure(fig_num())
    plt.title('Running of b value')
    plt.plot(X,b1,color = 'red')
    plt.plot(X,b2,color = 'blue')
    plt.plot(X,b3,color = 'green')
    plt.xlabel('Energy')
    plt.ylabel('b value')
    plt.xscale('log',basex = 10)
    plt.show()
   
    return

def remove_discontinuity(Y,threshold = 0.1):
    n = len(Y)
    
    for i in range(1,n):
        if abs(Y[i] - Y[i - 1]) > threshold:
            disc_jump = Y[i] - Y[i - 1]
            for j in range(i,n):
                Y[j] -= disc_jump
                
    return Y


def optimize_mass():
    n = 30
    i_check = [3,6,9,12,15,18,21,24,27]
    g = 4 # num of generations aka dimension of parameter space
    
    mass_min = 22680
    mass_max = 22690
    #mass_vec = np.geomspace(mass_min, mass_max,n)
    mass_vec = np.linspace(mass_min, mass_max,n)
    u_quality_vec = np.zeros(n)
    ones = np.ones(g)
    
    for i in range(n):
        u_quality_vec[i] = get_unification(mass_vec[i]*ones)
        if i in i_check:
            print('{} % complete...'.format(100*float(i)/n))

    
    idx_best_u_quality = np.argmin(u_quality_vec)
    mass_optimized = mass_vec[idx_best_u_quality]
    print('OPTIMIZED MASS: {}'.format(mass_optimized))
    print('UNIFICATION QUALITY = {}'.format(u_quality_vec[idx_best_u_quality]))
    
    title = 'OPTIMIZED MASS: {}, U QUALITY = {}'.format(mass_optimized,u_quality_vec[idx_best_u_quality])
    
    plt.figure()
    plt.plot(mass_vec,u_quality_vec)
    plt.title(title)
    plt.xlabel('Mass (GeV)')
    plt.ylabel('Unification Quality')
    plt.show()
    
    return mass_optimized
#######################################################################
#CLASSES
    
class Particle:
    
    def __init__(self,colour,su2_isospin,weak_hypercharge,chirality,mass):

        self.colour = colour # red,blue,green or None
        self.su2_isospin = su2_isospin # +/- 1/2
        self.weak_hypercharge = weak_hypercharge
        self.chirality = chirality # LH: -1, RH: +1
        self.mass = mass

#######################################################################
#MAIN
#mass_param = npa([14000,14000,14000,14000]) # mass for 4th, 5th, and 6th gen
#mass_param = 159178.922193*np.ones(4)
#mass_param = 22680*np.ones(4)
#mass_param = 3000*np.ones(4)
mass_param = 0*np.ones(4)
#mass_param = 23000*np.ones(4)

print('UNIFICATION QUALITY = {}'.format(get_unification(mass_param)))

#optimize_mass()






















