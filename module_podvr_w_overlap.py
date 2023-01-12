
"""
MODULE FOR GENERATING MULTIDIMENSIONAL BASIS & MULTIDIMENSIONAL POTENTIAL GRID 
MULTIDIMENSIONAL POTENTIAL GRID WILL BE OBTAINED FROM ONE DIMENSIONAL DVR GRIDS

"""

#########################################################################

import sys,math,random,copy
import numpy as np
from const_module import * 
from numpy import linalg as LA
from numpy.polynomial import hermite as hermite
from scipy.integrate import simps
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

#########################################################################

"""
FORMATION OF DIRECT PRODUCT GRID FROM ONE DIMENSIONAL DVR GRIDS 

"""

def DP_grid(A):
    """
    A is an array of arrays containing grid points for all the dimensions in the problem
    """
    A_length=len(A)
    if A_length==2:
        direct_product_grid_1=np.stack(np.meshgrid(A[0],A[1],indexing='ij'),A_length).tolist()
    elif A_length==3:
        direct_product_grid_1=np.stack(np.meshgrid(A[0],A[1],A[2],indexing='ij'),A_length).tolist()
    elif A_length==4:
        direct_product_grid_1=np.stack(np.meshgrid(A[0],A[1],A[2],A[3],indexing='ij'),A_length).tolist()
    elif A_length==5:
        direct_product_grid_1=np.stack(np.meshgrid(A[0],A[1],A[2],A[3],A[4],indexing='ij'),A_length).tolist()
    elif A_length==6:
        direct_product_grid_1=np.stack(np.meshgrid(A[0],A[1],A[2],A[3],A[4],A[5],indexing='ij'),A_length).tolist()
    elif A_length==7:
        direct_product_grid_1=np.stack(np.meshgrid(A[0],A[1],A[2],A[3],A[4],A[5],A[6],indexing='ij'),A_length).tolist()
    size=np.size(direct_product_grid_1)
    row=size/A_length
    direct_product_grid=np.reshape(direct_product_grid_1,(int(row),A_length)).tolist()
    ### returning as a python list .. with no. of rows equal to N1*N2*N3*....*Nd i.e. no. of elements in the direct product grid .. ###
    ### No.of columns equal to length of A .. i.e. no. of dimensions ###
    return direct_product_grid

############################################################################
"""
FUNCTION FOR GENERATING FULL-DIMENSIONAL CORRESPONDING TO DIRECT PRODUCT GRID FOR POTENTIAL CALCULATION 
"""

def generate_full_dim_pot_4_multidim_eigenstate(DP_grid,mode_index_arr):
    ### DP_grid is full direct product grid ###
    ### Length of mode_index_arr is equal to length of each element of DP_grid ###
    full_dim_grid=[]
    for i in range(len(DP_grid)):
        init_arr=np.zeros(33).tolist()
        for j in range(len(DP_grid[i])):
           init_arr[mode_index_arr[j]-1]+=DP_grid[i][j]  
        full_dim_grid.append(init_arr)
    return full_dim_grid

############################################################################

"""
FUNCTION FOR SORTING EIGENVECTOR MATRIX ACCORDING TO SORTING ORDER OF EIGENVALUE AFTER 
GETTING EIGENVALUE ARRAY FROM  NUMPY 
THE ARRAY RETURNED WILL BE AN ARRAY OF ARRAYS ..
i-TH ARRAY IS i-th EIGENVECTOR ..
"""

def eigvec_sorter(X,A):
    argsort=np.argsort(X).tolist()
    A_T=np.transpose(A).tolist()
    eigvec_rtrn=[]
    for i in range(len(argsort)):
        eigvec_rtrn.append(A_T[argsort[i]])
    return eigvec_rtrn

############################################################################

"""
CHECKING NORMALIZATION for 1-d dataset .. using simpson integration from scipy..
"""
def check_normal(x_arr,func_arr):
    Integral_val=simps(func_arr,x_arr)
    return Integral_val

############################################################################


def HO_val(x_dimless_val_arr,X_shift,N_HO):
    ## argument 'x_dimless_val_arr' gives the dvr grid where function values are to be calculated ##
    ## N_HO is no. of eigenfunctions for which values will have to be calculated ##
    x_dimless_val_shift_arr_1=[x_dimless_val_arr[k]+X_shift for k in range(len(x_dimless_val_arr))]
    N_length_herm=len(x_dimless_val_arr)            # no. of dvr points #
    hermite_index_arr=np.zeros(N_HO).tolist()
    HO_func_all_arr,HO_func_all_mod_sqr_arr=[],[]
    #######################################
    integral_arr=[]
    for i in range(N_HO):
        hermite_index_arr[i]+=1
        HO_func_cur_arr,HO_func_mod_sqr_cur_arr=[],[]
        for j in range(len(x_dimless_val_arr)):
            HO_func_cur=(1/(math.sqrt((2**i)*(math.factorial(i)))))*(1/(math.pi)**0.25)*hermite.hermval(x_dimless_val_arr[j],hermite_index_arr)*(math.exp(-(((x_dimless_val_arr[j])**2)/2)))
            HO_func_mod_sqr_cur=HO_func_cur**2
            HO_func_cur_arr.append(HO_func_cur)
            HO_func_mod_sqr_cur_arr.append(HO_func_mod_sqr_cur)
        HO_func_all_arr.append(HO_func_cur_arr)
        HO_func_all_mod_sqr_arr.append(HO_func_mod_sqr_cur_arr)
        hermite_index_arr[i]-=1
        integral_cur=check_normal(x_dimless_val_shift_arr_1,HO_func_mod_sqr_cur_arr)
        integral_arr.append(integral_cur)
    return HO_func_all_arr,HO_func_all_mod_sqr_arr,integral_arr

############################################################################

"""
FUNCTION FOR GENERATING ONE DIMENSIONAL PODVR BASIS BY SOLVING 1-D EFFECTIVE HAMILTONIANS
THESE 1-d EFFECTIVE HAMILTONIANS WILL BE SOLVED USING HO-DVR BASIS
"""

#################
"""
MULTIDIMENSIONAL WORKING BASIS WILL BE DIRECT PRODUCT OF 1-D PODVR BASIS FUNCTIONS ..
SCHEME FOR OBTAINING PODVR BASIS :
1. TAKING HO BASIS AS PRIMITIVE BASIS 
2. DIAGONALIZE 1-D COORDINATES USING THIS BASIS TO GET HO-DVR BASIS
3. USING HO-DVR FUNCTIONS TO SOLVE 1-D EFFCETIVE HAMILTONIANS .. GET NEW BASIS (DELOCALIZED) .. THESE ARE EIGENSTATES OF 1-D EFFECTIVE HAMILTONIANS
4. DIAGONALIZE COORDINATES USING THESE BASIS TO GET PODVR BASIS
"""

#####################################################################################

def HODVR_basis(X0,N_basis):
    """
    DIAGONALIZING X-MATRIX IN HO BASIS  .. GENERATING DVR POINTS ..
    POSITION SHIFTING WILL BE EFFECTED ACCORDINGLY ..
    DIMENSIONLESS X-MATRIX IS FREQUENCY INDEPENDENT
    KE MATRIX ELEMENTS ARE INVARIANT W.R.T. TRANSLATION ..DUE TO INFINITE BOUNDARY ..
    FBR_2_DVR TRANSFORMATION MATRIX WILL ALSO BE INVARIANT W.R.T. TRANSLATION ..
    """
    x_mat=np.zeros((N_basis,N_basis)).tolist()
    for i in range(len(x_mat)):
        for j in range(len(x_mat[i])):
            if i==j+1:
                x_mat[i][j]+=(1/(math.sqrt(2)))*(math.sqrt(j+1))
            elif i==j-1:
                x_mat[i][j]+=(1/(math.sqrt(2)))*(math.sqrt(j))
    x_mat_non_diag=copy.deepcopy(x_mat)     ## x-matrix before diagonalization ##
    x_eigvals,x_eigvecs=LA.eig(x_mat)
    x_eigvals_sort=np.sort(x_eigvals).tolist()
    x_sodvr_pts=[x_eigvals_sort[k]+X0 for k in range(len(x_eigvals_sort))]
    x_eigvecs_sort=eigvec_sorter(x_eigvals,x_eigvecs)
    ########################################################
    ########## constructing fbr_2_dvr transformation matrix ###########
    fbr_2_dvr=np.zeros((N_basis,N_basis)).tolist()
    hermite_index=np.zeros(N_basis).tolist()
    a,b=hermite.hermgauss(N_basis)
    a1,b1=a.tolist(),b.tolist()
    for i in range(len(fbr_2_dvr)):
        for j in range(len(fbr_2_dvr[i])):
            hermite_index[j]+=1
            fbr_2_dvr[i][j]=(math.sqrt(b1[i]))*(hermite.hermval(a1[i],hermite_index))*(1/((math.pi)**0.25))*(1/(math.sqrt((2**j)*math.factorial(j))))
            hermite_index[j]-=1
    ################################################
    """ 
    Writing loop for returning values of [(wj/w(xj))^0.5] for all dvr functions at its own DVR point  
    """
    weight_scl_arr=[]           ### scale factor multiplying which gives unity at peak for dvr basis functions ###
    for i in range(len(x_sodvr_pts)):
       weight_scl_arr.append(math.sqrt(b1[i]/(math.exp(-((x_sodvr_pts[i]-X0)**2))))) 
    ##### returning Potential values at the shifted oscillator dvr points .. & fbr_2_dvr points ########
    ##### fbr_2_dvr is the eigenvector matrix .. each row is an eigenvector #######
    return x_sodvr_pts,fbr_2_dvr,weight_scl_arr

##################################################################################
##################################################################################
##################################################################################

"""
GEENERATING PODVR BASIS FROM HODVR BASIS
POTENTIAL PART WILL BE CALCULATED ON THE MULTIDIMENSIONAL GRID POINTS BEFOREHAND AND WILL 
BE GIVEN IN THE ARGUMENT IN THE FUNCTION THAT GENERATE PODVR BASIS

"""

def anharm_eigvecs(V_HODVR_arr,FBR_2_DVR,freq):    
    """
    V_HODVR_arr is an array with potential values at the SODVR points
    FBR_2_DVR is a matrix for converting KE part from FBR to DVR basis 
    Point To be noted ..
    freq used would be the frequency for the corresponding Normal mode defined at the TS ..
    As Kinetic Energy part would be insensitive to the different potential cuts 
    """
    N_basis=len(FBR_2_DVR)				# No. of Shifted Oscillator primitive basis #
    V_anharmonic=np.diag(V_HODVR_arr).tolist()		# generating diagonal potential matrix #
    ##################################################################
    """
    GENERATING KE MATRIX IN FBR BASIS AND THEN TRANSFORMING IN TERMS OF DVR BASIS
    T_freq will be KE matrix where freq. information in incorporated and will be useful for single minimum problem
    """
    T_freq=np.zeros((N_basis,N_basis)).tolist()
    for i in range(len(T_freq)):
        for j in range(len(T_freq[i])):
            if i==j:
                T_freq[i][j]+=freq*((j/2)+0.25)
            elif i==j-2:
                T_freq[i][j]-=(freq/4)*(np.sqrt(j*(j-1)))
            elif i==j+2:
                T_freq[i][j]-=(freq/4)*(np.sqrt((j+1)*(j+2)))
    FBR_2_DVR_mat=np.matrix(FBR_2_DVR)
    FBR_2_DVR_T=np.transpose(FBR_2_DVR)
    T_FBR_mat_freq=np.matrix(T_freq)
    T_DVR_freq=np.matrix(np.dot(FBR_2_DVR_mat,(np.matmul(T_FBR_mat_freq,FBR_2_DVR_T))))           # KE matrix in SODVR basis #
    """
    GETTING THE FULL HAMILTONIAN FOR THE ANHARMONIC POT FOR THE GIVEN CUT IN DVR
    DIAGONALISING AND SAVING BASIS OF THE ANHARMONIC POTENTIAL ..
    """
    V_anharm_mat=np.matrix(V_anharmonic)
    H_anharm_cut=np.add(T_DVR_freq,V_anharmonic)				# Hamiltonian matrix of the anharmonic oscillator #
    E_anharm_eigvals_1,E_anharm_eigvecs_1=LA.eig(H_anharm_cut)
    E_anharm_eigvals_sorted=np.sort(E_anharm_eigvals_1).tolist()					# Sorted Eigenvalues of the Anharmonic Oscillator #
    E_anharm_eigvecs=eigvec_sorter(E_anharm_eigvals_1,E_anharm_eigvecs_1)	# Saving Eigenvectors of the Anharmonic Oscillator in terms of SODVR basis #
    """
    i-th array of Anharm_basis corresponds to i-th eigenvector .. the elements of that array are coeffs for i-th eigenvector
    """
    Anharm_basis_chk=copy.deepcopy(E_anharm_eigvecs)
    #######################
    ####### Returns Eigenvalues & Eigenvectors of Anharmonic oscillator cut ; KE matrix in SODVR basis (T_DVR_freq) ######
    return E_anharm_eigvals_sorted,Anharm_basis_chk,T_DVR_freq

######################################################################################
######################################################################################
######################################################################################

"""
FUNCTION FOR GENERATING PODVR POINTS AND PODVR BASIS FROM DELOCALIZED FUNCTIONS i.e. EIGENFUNCTIONS OF 
ANHARMONIC OSCILLATOR CORRESPONDING TO A PARTICULAR DIMENSION FOR A GIVEN CUT

.....POINT TO BE NOTED ....
EIGENFUNCTIONS OF THE ANHARMONIC OSCILLATOR CUT IN TERMS OF SODVR BASIS (SHIFTED OSCILLATOR DVR)
SODVR_BASIS..SO_EIGVECS..PODVR_BASIS
"""

def podvr_basis(anharm_eigvec,X_DVR):
    """
    EIGENFUNCTIONS OF THE CONCERNED ANHARMONIC OSCILLATOR IS OBTAINED IN TERMS OF HO-DVR BASIS
    anharm_eigvec is coeffs in terms of SO-DVR basis .. only those vectors are retained that converged ..
    ###=== i-th row of anharm_eigvec is coeffs for SODVR basis .. ===###
    ###=== No. of rows < No. of columns in anharm_eigvec matrix since only a subset of eigenvectors are retained ===###
    ### anharm_eigvec_all is a square matrix that transforms from SODVR basis => Anharm_eigenbasis ###
    X_DVR is diagonal X-matrix in terms of SO-DVR basis .. Size of X_DVR matrix is N_prim i.e. size of primitive basis 
    """
    N_anharm_basis=len(anharm_eigvec)
    X_anharm=np.zeros((N_anharm_basis,N_anharm_basis)).tolist()		# Initialising X-matrix in PO basis #
    for i in range(len(anharm_eigvec)):
        for j in range(len(anharm_eigvec)):
            a=0.
            for k in range(len(anharm_eigvec[i])):
                a+=(anharm_eigvec[i][k]*anharm_eigvec[j][k])*X_DVR[k][k]
            X_anharm[i][j]+=a
    zz1,zz2=LA.eig(X_anharm)
    zz3=eigvec_sorter(zz1,zz2)
    PODVR_pts_sorted=np.sort(zz1).tolist()  # Sorting PODVR points #
    PODVR_basis=copy.deepcopy(zz3)          # PODVR basis in terms of eigenfunctions of anharmonic oscillator #
    #########==========================#############
    """
    PODVR basis is written in terms of eigenfunctions of the anharmonic oscillator .. i.e. the oscillator generated after 
    taking the cut
    """
    return PODVR_pts_sorted,PODVR_basis

########################################################################
########################################################################
########################################################################

"""
FUNCTION FOR GENERATING FULL DIMENSIONAL GRID CORRESPONDING TO THE SO_DVR POINTS
INPUT ARRAY OF LENGTH 3N-6 ; VALUES OF OTHER COORDINATES IN THE GIVEN CUT IS GIVEN
KEEPING THE RELEVANT COORDINATE VALUE(FOR WHICH SODVR IS BEING EVALUATED)
COORD_INDEX DENOTE THE RELEVANT COORDINATE AND X_SODVR DENOTE THE DVR POINTS
"""
def generate_full_grid_podvr_cut(full_grid_cut,coord_index,X_SODVR):
    sodvr_f_grid=[]
    for i in range(len(X_SODVR)):
        full_grid_cut_1=copy.deepcopy(full_grid_cut)
        full_grid_cut_1[coord_index-1]+=X_SODVR[i]
        sodvr_f_grid.append(full_grid_cut_1)
    return sodvr_f_grid

###################################

"""
FUNCTION FOR CHECKING STATUS REGARDING CONVERGENCE CRITERIA 
"""
def check_for_greater(A_arr,chk_arr):
    """
    A_arr is array for which checking is to be done;
    chk_arr is array containing equal no of values containing
    convergence criteria for each of the element in A_arr
    """
    ctr=0
    for i in range(len(A_arr)):
        if A_arr[i]>chk_arr[i]:
            ctr+=1
    if ctr!=0:
        d=str("false")
    else:
        d=str("true")
    return d

####################################################################
####################################################################
####################################################################
"""
TEST POTENTIAL FUNCTION FOR CHECKING THE CODE

"""

def test_pot(x):
	v1=0.
	v1+=(5.726/(1.239848738e-4))*((math.exp(-(2.44*(x*0.17431991))))-2*(math.exp(-((1.22*(x*0.17431991))))))
	return v1

####################################################################


"""
FUCNTION FOR GETTING HARMONIC OSCILLATOR CORRESPONDING TO A MINIMUM AFTER LOCATING THE MINIMUM
"""
def HO_minima(cur_coord_grid,E_cur_grid):
    delta_4_numer_drv_arr=np.diff(cur_coord_grid)
    delta_4_numer_drv=np.mean(delta_4_numer_drv_arr)
    cur_coord_max,cur_coord_min=cur_coord_grid[len(cur_coord_grid)-1],cur_coord_grid[0]             # 1st element of the array is the minimum, last element is the maximum #
    E_min_val,E_min_index_cur=np.amin(E_cur_grid),np.argmin(E_cur_grid)                             # Value of minimum of energy & location in the array #
    E_vals_4_freq=[E_cur_grid[E_min_index_cur-4],E_cur_grid[E_min_index_cur-3],E_cur_grid[E_min_index_cur-2],E_cur_grid[E_min_index_cur-1],E_cur_grid[E_min_index_cur],E_cur_grid[E_min_index_cur+1],E_cur_grid[E_min_index_cur+2],E_cur_grid[E_min_index_cur+3],E_cur_grid[E_min_index_cur+4]]
    diag_2nd_drv_Energy=(1/delta_4_numer_drv**2)*(((-0.178571428571433E-02)*(E_vals_4_freq[0]))+((0.253968253968258E-01)*(E_vals_4_freq[1]))+((-0.200000000000000E+00)*E_vals_4_freq[2])+((0.160000000000000E+01)*(E_vals_4_freq[3]))+((-0.284722222222222E+01)*(E_vals_4_freq[4]))+((0.160000000000000E+01)*(E_vals_4_freq[5]))+((-0.200000000000000E+00)*(E_vals_4_freq[6]))+((0.253968253968258E-01)*(E_vals_4_freq[7]))+((-0.17857142571433E-02)*(E_vals_4_freq[8])))
    ##### creating difference array for all grid points from the minima location #######
    cur_diff_grid_arr=[]
    E_harmonic_cur=[]               # generating corresponding HO energies about the minimum #
    X_min_val_cur=cur_coord_grid[E_min_index_cur]       # value of the grid point where energy minimum is before extending the grid left or right as required  #
    ###################################################
    ##### generating symmetric grid for HO generation ##############
    HO_generate_grid=copy.deepcopy(cur_coord_grid)
#        index_cur_coord_max=cur_coord_grid.index(cur_coord_grid[len(cur_coord_grid)-1])          # array index of maximum i.e. last element of cur_coord_grid #
    diff_index_max_E_min=(len(cur_coord_grid))-1-E_min_index_cur         # difference between indices of the maximum element & element where energy minimum is #
    if diff_index_max_E_min>E_min_index_cur:
        diff_rqur_left=diff_index_max_E_min-E_min_index_cur                  # No. of elements to be included to the left #
        for i in range(1,diff_rqur_left+1):
            insert_cur=cur_coord_grid[0]-i*delta_4_numer_drv
            HO_generate_grid.insert(0,insert_cur)
            E_min_index_cur+=1
    elif diff_index_max_E_min<E_min_index_cur:
#        index_cur_coord_min=cur_coord_grid.index(cur_coord_grid[0])                             # array index of minimum element of cur_coord_grid #
        diff_rqur_right=E_min_index_cur-(len(cur_coord_grid)-1-E_min_index_cur)    # No. of elements to be included to the right #
        for i in range(1,diff_rqur_right+1):
            insert_cur=cur_coord_grid[len(cur_coord_grid)-1]+i*delta_4_numer_drv
            HO_generate_grid.append(insert_cur)
    for i in range(len(HO_generate_grid)):
        cur_diff_grid_arr.append(HO_generate_grid[i]-HO_generate_grid[E_min_index_cur])
        E_harmonic_cur.append(0.5*diag_2nd_drv_Energy*((HO_generate_grid[i]-HO_generate_grid[E_min_index_cur])**2)+E_min_val) 
#    print(E_harmonic_cur)
#    fig1,ax=plt.subplots()
#    plt.plot(HO_generate_grid,E_harmonic_cur,linestyle='--',color='green',marker='o',label='harmonic')
#    ax.set_xlabel("Q$_{scaled}$",fontsize=15,fontweight='bold')
#    ax.set_ylabel('Energy(cm$^{-1}$',fontsize=15,fontweight='bold')
#    plt.legend(loc='best')
#    plt.show()
    return diag_2nd_drv_Energy,E_min_val,X_min_val_cur


#########################################################################

"""
FUNCTION FOR PLOTTING EIGENFUNCTIONS IN 1 DIMENSION AND 2 DIMENSIONAL PROJECTION OF MULTIDIMENSIONAL EIGENFUNCTION 
"""

def eigenfunc_plot(coeff_arr,basis_val_arr):
    """
    Multidimensional Grid .. array of arrays .. each element of multidimensional grid contains elements equal to N_dim .. no of dimensions of the problem ..
    basis_val_arr also an array of arrays .. each element gives values of all basis functions at a grid point ..
    length of basis_val_arr is equal to no of grid points .. length of each element of basis_val_arr is equal to no. of basis functions
    ##########################
    coeff_arr is also an array of arrays .. length of coeff_arr is equal to no of eigenfunctions ..
    length of each element in coeff_arr is equal to no. of basis functions .. 
    ##########################
    output is eigfunc_all_arr is also an array of arrays .. length of eigfunc_all_arr is equal to no of eigenfunctions ..
    length of each element in eigfunc_all_arr is equal to no of grid points ..  
    """
    eigfunc_all_arr=[]                  # array of arrays .. each element gives values of an eigenfunction at all grid points #
    eigpeak_all_arr=[]
    for i in range(len(coeff_arr)):
        eigfunc_arr=[]
        for j in range(len(basis_val_arr)):
            eigfunc_at_cur_grid=0.
            for k in range(len(coeff_arr[i])):
                eigfunc_at_cur_grid+=coeff_arr[i][k]*basis_val_arr[j][k]
            eigfunc_arr.append(eigfunc_at_cur_grid)
        eigpeak_all_arr.append(np.amax(eigfunc_arr))          ## peak value of current eigenfunction ##
        eigfunc_all_arr.append(eigfunc_arr)
    return eigfunc_all_arr,eigpeak_all_arr

###########################################################################
###########################################################################

"""
Function for calculating difference in an array of arrays starting from the last element & plotting difference w.r.t. basis size
"""

def plot_diff_eig_conv(N_basis_min,E_eigvals_arr):
    N_basis_last=N_basis_min+len(E_eigvals_arr)-1           ## Value of Max. Basis tried out ##
    Eigvals_subtract_arr=[]
    Eigvals_diff_ref=E_eigvals_arr.pop()
    for i in range(len(E_eigvals_arr)):
        x=np.subtract(Eigvals_diff_ref,E_eigvals_arr[i]).tolist()
        Eigvals_subtract_arr.append(x)
    Eigvals_subtract_arr_t=np.transpose(Eigvals_subtract_arr).tolist()          ## each array gives energy difference w.r.t. ref basis size for a particular state ##
    ##### Creating x-array w.r.t which plotting will be done #####
    x_arr=[k+N_basis_min for k in range(len(E_eigvals_arr))]    
    ##############################################################
    fig,ax=plt.subplots()
    for i in range(10):
        plt.plot(x_arr,Eigvals_subtract_arr_t[i],linestyle='-',marker='o',label=i+1)
        ax.set_xlabel("Basis Size",fontsize=15,fontweight='bold')
        ax.set_ylabel("Difference from Energy from ref basis",fontsize=15,fontweight='bold')
        ax.set_ylim(-0.005,0.005)
        plt.legend(loc='best')
    plt.title("Convergence Checking for first 10 States",fontsize=15,fontweight='bold')
    plt.show()
    fig1,ax1=plt.subplots()
    for i in range(10,len(Eigvals_subtract_arr_t)):
        plt.plot(x_arr,Eigvals_subtract_arr_t[i],linestyle='-',marker='o',label=i+1)
        ax1.set_xlabel("Basis Size",fontsize=15,fontweight='bold')
        ax1.set_ylabel("Difference from Energy from ref basis",fontsize=15,fontweight='bold')
        ax1.set_ylim(-1.005,1.005)
        plt.legend(loc='best')
    plt.title("Convergence Checking for last 5 States",fontsize=15,fontweight='bold')
    plt.show()
    return x_arr,Eigvals_subtract_arr_t

###########################################################################
###########################################################################

"""
FUNCTION FOR GENERATING SINC- DVR POINTS AND BASIS FUNCTIONS  
"""

def sinc_dvr(N_global_min,x_min,x_max,X_grid,N_dvr,freq):
    """
    N_global_min is no. of global minima present in the 1-d cut
    x_conversion_fact is conversion factor for converting normal coord from (amu)**(1/2) A^0  => dimless
    x_min,x_max will define the range of the dvr grid ..
    N_dvr will be an input that denotes number of dvr basis functions .. no. of dvr points 
    freq is the frequency of the corresponding Normal Mode 
    """
    ###########################
    h_amu_ang_sqr=h_J*(1e+20)*(1/amu)                       ## Planck constant in amu ang**2 s**-1 ##
    h_bar_amu_ang=h_amu_ang_sqr/(2*(np.pi))                 ## h_bar in amu ang**2 s**-1 ##
    x_conversion_fact=0.17222125670294633*(np.sqrt(freq))       ## conversion from dimensional to dimsionless units  for the corresponding Normal Mode  ##
    ###########################
    delta_dimless=(x_max-x_min)/N_dvr                       ## spacing of dimensionless dvr grid ##
    x_min_dim,x_max_dim=x_min/x_conversion_fact,x_max/x_conversion_fact
    delta_dim=(x_max_dim-x_min_dim)/N_dvr                   ## spacing of dimensional dvr grid ##
    Basis_size=N_dvr+1
    dvr_grid_dimless=[x_min+j*delta_dimless for j in range(Basis_size)]
    ##########################################################
    """
    Returning values of sinc-dvr basis functions in coordinate grid .. i.e. X_grid for plotting purpose
    """
    sinc_dvr_all=[]
    for i in range(Basis_size):
        sinc_dvr_cur=[]         # values of current sinc-dvr basis function at all the X_grid points #
        for j in range(len(X_grid)):
            if X_grid[j]!=dvr_grid_dimless[i]:
                sinc_dvr_cur.append((math.sin(((math.pi)/delta_dimless)*(X_grid[j]-dvr_grid_dimless[i])))/(((math.pi)/delta_dimless)*(X_grid[j]-dvr_grid_dimless[i])))
            else:
                sinc_dvr_cur.append(1.)
        sinc_dvr_all.append(sinc_dvr_cur)
    """
    evaluating KE matrix elements in sinc-dvr basis 
    """
    KE_sinc_dvr_dimless=np.zeros((Basis_size,Basis_size)).tolist()           ## Initialising KE matrix dimensionless dvr##
    KE_sinc_dvr_dim=np.zeros((Basis_size,Basis_size)).tolist()           ## Initialising KE matrix dimensional dvr ##
    for i in range(len(KE_sinc_dvr_dimless)):
        for j in range(i,len(KE_sinc_dvr_dimless[i])):
            if j==i:
                KE_sinc_dvr_dimless[i][j]=(0.5*freq)*(1/3)*(((math.pi)**2)/(delta_dimless**2))
                KE_sinc_dvr_dim[i][j]=(0.5*(h_bar_amu_ang**2))*(1/3)*(((math.pi)**2)/(delta_dim**2))
            else:
                KE_sinc_dvr_dimless[i][j]=(0.5*freq)*(2/delta_dimless**2)*(((-1)**(i-j))/((i-j)**2))
                KE_sinc_dvr_dim[i][j]=(0.5*(h_bar_amu_ang**2))*(2/delta_dim**2)*(((-1)**(i-j))/((i-j)**2))
                KE_sinc_dvr_dimless[j][i]=KE_sinc_dvr_dimless[i][j]
                KE_sinc_dvr_dim[j][i]=KE_sinc_dvr_dim[i][j]
    #################################################################
    #### Unit conversion for KE matrix for dimensional units .. converting from (amu ang**2 s**-2) to cm**-1 ####
    #### This is done by dividing by (hc) where 'h' is in amu ang**2 s**-1 & 'c' is in cm s**-1 units ####
    v_l=100*c       ### speed of light in cm s**-1 ###
    KE_sinc_dvr_wn=[[KE_sinc_dvr_dim[i][j]/(h_amu_ang_sqr*v_l) for j in range(len(KE_sinc_dvr_dim[i]))] for i in range(len(KE_sinc_dvr_dim))]
    if N_global_min==1:
        return dvr_grid_dimless,sinc_dvr_all,KE_sinc_dvr_dimless
    else:
        return dvr_grid_dimless,sinc_dvr_all,KE_sinc_dvr_wn

################################################################################
################################################################################
"""
WRITING FUNCTION FOR CHOICE OF DVR .. i.e. Sinc-DVR or HO-DVR 

"""

###############
###############

def dvr_pick(N_global_min):
    """
    N_global_min gives number of global minima 
    """
    if N_global_min>1:
        return str("sinc")
    else:
        return str("HO")

###############################################################################
###############################################################################

"""
Building KE matrix  .. in DPB basis 
"""
class tensor_product_left_right:
    """
    Kinetic Energy Operator matrix DVR Discretized
    The Full Kinetic Energy Matrix is generated as a summation of matrices generated from direct product 
    For all dimensions the operations are similar .. except for the location of the 1-d KE mat for the relevant coordinate
    The dataset needed is also similar for all coordinates 
    """
    ### Initializer ###
    def __init__(self,cur_index,coord_basis_size_arr):
        self.N_dim=len(coord_basis_size_arr)                        ## length of coord_basis_size_arr gives no of dimensions considered in the calculation ## 
        self.current_index=cur_index                                ## current index denotes the index for which the direct product structure will be constructed ## 
        self.basis_size_arr=coord_basis_size_arr                    ## An array containing basis size numbers for different coordinates considered in the calculation ##
    def direct_product(self):
        """
        This function will return direct product of identity matrices left and right ..
        i.e. just remains .. direct product of the KE matrix for the relevant coordinate with 
        the L_DP & R_DP .. left & right direct product of identity matrices ..
        """
        R_identity_no,L_identity_no=(self.N_dim-self.current_index),(self.current_index-1)         ## no. of identity matrices for direct product in the left and right of the current index ##
        ##### Generating Left & Right direct product identity matrices ######
        #####################################################################
        ####### LEFT #######
        if L_identity_no==0:
            x_left_dpb_prv=str("None")
        elif L_identity_no==1:
            x_left_dpb_prv=np.eye(self.basis_size_arr[0]).tolist()                          
        else:
            x_left_dpb_prv=np.eye(self.basis_size_arr[0]).tolist()                          ## Initialising for generating full left identity for direct product ##
            for j in range(1,L_identity_no):
                x_left_dpb_cur=np.kron(x_left_dpb_prv,(np.eye(self.basis_size_arr[j]))).tolist()
                x_left_dpb_prv=x_left_dpb_cur
        ##########################################################################
        ######## RIGHT #######
        if R_identity_no==0:
            x_right_dpb_prv=str("None")
        elif R_identity_no==1:
            x_right_dpb_prv=np.eye(self.basis_size_arr[self.current_index]).tolist()
        else:
            x_right_dpb_prv=np.eye(self.basis_size_arr[self.current_index]).tolist()                             ## Initialising for right dpb generation ##
            for j in range(self.current_index+1,self.N_dim):
                x_right_dpb_cur=np.kron(x_right_dpb_prv,np.eye(self.basis_size_arr[j])).tolist() 
                x_right_dpb_prv=x_right_dpb_cur
        return x_left_dpb_prv,x_right_dpb_prv
    def KE_PODVR_basis(self,KE_SODVR,A_eigvecs_sodvr_2_anharm,A_eigvecs_anharm_2_podvr):
        """
        KE_SODVR_mat=KE_SODVR                                  ## KE matrix for the relevant coordinate in SODVR Basis (either from HODVR or from Sinc-DVR)
        sodvr_2_anharm=A_eigvecs_sodvr_2_anharm                ## Eigenvector Matrix for the Anharmonic Oscillator Cut ##
        anharm_2_podvr=A_eigvecs_anharm_2_podvr                ## Eigenvector Matrix that diagonalises X-matrix ##
        """
        N_anharm_basis=len(A_eigvecs_sodvr_2_anharm)            ## length of eigenvector matrix for sodvr => anharm eigenstates ##
        KE_anharm_eigvec=np.zeros((N_anharm_basis,N_anharm_basis)).tolist()
        for i in range(N_anharm_basis):
            for j in range(N_anharm_basis):
                s1=0.
                for m in range(len(A_eigvecs_sodvr_2_anharm[i])):
                    for n in range(len(A_eigvecs_sodvr_2_anharm[j])):
                        s1+=(A_eigvecs_sodvr_2_anharm[i][m]*A_eigvecs_sodvr_2_anharm[j][n])*KE_SODVR[m][n]
                KE_anharm_eigvec[i][j]+=s1
        KE_podvr=np.matrix(np.dot(A_eigvecs_anharm_2_podvr,(np.matmul(KE_anharm_eigvec,(np.matrix(A_eigvecs_anharm_2_podvr).T.tolist()))))).tolist()
        return KE_podvr
    def KE_DPB(self,KE_SODVR,A_eigvecs_sodvr_2_anharm,A_eigvecs_anharm_2_podvr):
        """
        returns direct product structure for the current coordinate 
        """
        x1,x2=self.direct_product()
        x3=self.KE_PODVR_basis(KE_SODVR,A_eigvecs_sodvr_2_anharm,A_eigvecs_anharm_2_podvr)
        if x1=="None":
            dpb_2=np.kron(x3,x2).tolist()
        elif x2=="None":
            dpb_2=np.kron(x1,x3).tolist()
        else:
            dpb_1=np.kron(x1,x3).tolist()
            dpb_2=np.kron(dpb_1,x2).tolist()
        return dpb_2


#####################################################################################

"""
Functions for multidim. index for DPB from individual 1-d indices and getting 1-d indices from multidim 
indices .. given only basis size for all dimensions 
"""
class INDEX_ONE_2_MULTIDIM_DPB_BACK_AND_FORTH:
    """
    FINDING MULTIDIM. INDEX FOR DIRECT PRODUCT BASIS FROM 1-D INDICES ..
    FINDING INDIVIDUAL INDICES FROM MULTIDIM INDEX FOR THE DIRECT PRODUCT BASIS
    """
    def __init__(self,basis_size_arr):
        self.basis_size_arr=basis_size_arr              ## array containing basis size for individual coordinates ##
        self.N_dim=len(basis_size_arr)                  ## no of coordinates considered in eigenstate calculation ##
    ##----------------------------------------------##
    def stride_arr(self):
        """
        preparing stride array
        """
        cur_index_pdt=1
        for j in range(1,self.N_dim):
            cur_index_pdt*=self.basis_size_arr[j]
        stride_arr_init=[cur_index_pdt]                 ## initializing stride array with first element ## 
        ##------------------------------##
        """ other elements of stride array will be prepared from the first element of the stride array """
        cur_index_init=1                                ## initializing current index for generating other elements of stride array ##
        while True:
            if cur_index_init==(self.N_dim-1):
                break
            else:
                cur_product=(int(cur_index_pdt/self.basis_size_arr[cur_index_init]))
                cur_index_init+=1
                stride_arr_init.append(cur_product)
                cur_index_pdt=copy.deepcopy(cur_product)
        return stride_arr_init
    ##---------------------------------------------##
    def multidim_index_DPB(self,one_dim_index_arr):
        """ given one dimensional indices , returns multidimensional basis no """
        ### one_dim_index_arr has python indexing .. zero based indexing ###
        stride_arr=self.stride_arr()            ## calling stride array ##
        multidim_basis_index=one_dim_index_arr[len(one_dim_index_arr)-1]+1
        for i in range(len(stride_arr)):
            multidim_basis_index+=stride_arr[i]*one_dim_index_arr[i]
    ### --------- returning multidim basis index .. multidim index has one-based indexing ---------- ###
        return multidim_basis_index
    ##-------------------------------------------------------##
    def one_dim_indices(self,multidim_index):
        """ given multidim index for multidim DPB, returns individual one dimensional indices """
        ### ``` multidim_index ``` indexing starts from 1 ### 
        stride_arr=self.stride_arr()            ## generating object ##
        multidim_index_4_caln=multidim_index-1  ## subtracting 1 that is the last term in the sum to go from 1-d indices to final multidim index ##
        onedim_index_arr=[]
        multidim_index_cur=copy.deepcopy(multidim_index_4_caln)     ## multidim_index will change for finding each of the 1-d indices in the loop .. here it is first initialized ##
        for i in range(len(stride_arr)):
            cur_onedim_index=multidim_index_cur//stride_arr[i]
            onedim_index_arr.append(cur_onedim_index)
            multidim_index_cur-=cur_onedim_index*stride_arr[i] 
        onedim_index_arr.append(multidim_index_cur)
        ##-- Returns 1-dimensional index array .. zero based indexing -- ##
        return onedim_index_arr


#####################################################################################

""" Plotting in 2-d .. of potentially multidimensional functions """

###---------###
""" 
Direct product grid/basis is prepared using podvr points/basis from individual 1-d calculations ..
Each DPB index starting from` 1` will be used to generate corresponding 1-d indices ..
a routine will be written for giving podvr function values at all grid points for all podvr functions for all dimensions
"""

###------------------------------------------------------------------------##
""" 
In any routine where eigenvector coeff. is used .. it has to be sorted in order of eigenvalues 
I-th row of the matrix gives coeff for I-th eigenvector 
"""


class wavefunc_multidim_plot(INDEX_ONE_2_MULTIDIM_DPB_BACK_AND_FORTH):
    """ 
    Inherits attributes of class INDEX_ONE_2_MULTIDIM_DPB_BACK_AND_FORTH
    """
    def __init__(self,basis_size_arr,DPB_size,mode_index_arr,dvr_type_arr,Q_shift_arr,plt_coord_1_arr,plt_coord_2_arr,sodvr_2_anharm_all_arr,anharm_2_podvr_all_arr,sodvr_pts_all_arr):
        super().__init__(basis_size_arr)          ## using 'super' to inherit variables defined in __init__ in parent class ##
        ###---- adding extra variables specific to this class ----###
        self.multidim_index=DPB_size                            ## size of direct product basis ##
        self.mode_index_arr=mode_index_arr                      ## array for mode indices (TS normal mode index) for the eigenstate calculation .. length equal to N_dim ##
        self.dvr_type_arr=dvr_type_arr                          ## array of strings containing either 'HO' or 'sinc' in order of mode index array ##
        self.Q_shift_arr=Q_shift_arr                            ## Q_shift_arr is array for coord shift where SODVR is used .. in order of mode indices 
        self.plt_q_1st_min,self.plt_q_1st_max,self.plt_q_1st_N_pts=plt_coord_1_arr[0],plt_coord_1_arr[1],plt_coord_1_arr[2]
        self.plt_q_2nd_min,self.plt_q_2nd_max,self.plt_q_2nd_N_pts=plt_coord_2_arr[0],plt_coord_2_arr[1],plt_coord_2_arr[2]
    ###-- eigenvector transformation matrices are saved as an array of matrices .. in sequence of normal mode indices taken for eigenstate calculation--### 
        self.sodvr_2_anharm_eigvecs=sodvr_2_anharm_all_arr        ## eigenvector matrices sodvr basis => anharmonic eigenvector basis ##
        self.anharm_2_podvr_eigvecs=anharm_2_podvr_all_arr        ## anharmonic eigenvector basis => podvr basis ##
        self.sodvr_pts_all_arr=sodvr_pts_all_arr                ## array containing sodvr(primitive-dvr) pts for all coords ## 
    ##---------------------------------------------------------------------##
    """ Transformation matrices are saved according to mode_index_arr order """
    ###--- function for generating one-dimensional index array for all multidimensional basis index -- ##
    def get_1_dim_index_arr(self):
        """
        self.multidim_index is equal to length of direct product grid ..i.e. size of direct product basis 
        """
        onedim_index_arr=[]
        for i in range(1,self.multidim_index+1):
            cur_onedim_index=super().one_dim_indices(i) 
            onedim_index_arr.append(cur_onedim_index)
        ##-- returns one-dimensional index array for all basis in the direct product basis -- ##
        return onedim_index_arr
    """
    ###--- function for giving values of 1-d  podvr functions at a grid point -- ###
    """
    def prim_basis_val(self,N_HO,Q_cur_val):
        """
        N_HO is number of HO functions for which values are to be calculated in the entire grid
        """
#        mode_index_id=self.mode_index_arr.index(mode_index)         ## python index for the concerned mode ##
#        N_HO=len(self.sodvr_2_anharm_eigvecs[mode_index_id])         ## length of this dvr=> anharm_eigvecs matrix gives no of primitive basis ##
        hermite_index_arr=np.zeros(N_HO).tolist()
        HO_func_all_at_cur_grid=[]                                  ## getting values of all HO functions served as primitive basis at the current configuration ##
        for i in range(N_HO):
            hermite_index_arr[i]+=1
            HO_cur_func_at_grid=(1/(math.sqrt((2**i)*(math.factorial(i)))))*(1/(math.pi)**0.25)*hermite.hermval(Q_cur_val,hermite_index_arr)*(math.exp(-(((Q_cur_val)**2)/2)))
            HO_func_all_at_cur_grid.append(HO_cur_func_at_grid)
            hermite_index_arr[i]-=1
        ##-----------------------------------------------------##
        ## -- for returning values of HO values .. coord shift value is not needed -- ##
        return HO_func_all_at_cur_grid
    ##-----------------------------------------------------------##
    def dvr_basis_val_arr(self,mode_index,cur_coord_val):
        """
        function for returning values of all dvr functions (N_prim for the concerned coord) at the current coord value
        if for a coordinate sinc-dvr is used .. shift value is kept at zero for that coordinate 
        converged 
        """
        mode_index_id=self.mode_index_arr.index(mode_index)         ## python index for the concerned mode ##
        cur_dvr_type=self.dvr_type_arr[mode_index_id]               ## getting dvr type `HO` / `sinc` for the current coordinate ##         
        dvr_pts_cur=self.sodvr_pts_all_arr[mode_index_id]                ## getting 1-d dvr points for the relevant coordinate ##
        N_prim=len(self.sodvr_2_anharm_eigvecs[mode_index_id][0])   ## No. of primitive basis used ##
        if cur_dvr_type=="sinc":
            delta=(dvr_pts_cur[len(dvr_pts_cur)-1]-dvr_pts_cur[0])/(len(dvr_pts_cur)-1)
            sinc_dvr_all_at_cur_grid=[]     ## array for storing values of all sinc-dvr function values at the current grid ##
            for i in range(N_prim):
                if cur_coord_val!=dvr_pts_cur[i]:
                    sinc_dvr_all_at_cur_grid.append((math.sin(((math.pi)/delta)*(cur_coord_val-dvr_pts_cur[i])))/(((math.pi)/delta)*(cur_coord_val-dvr_pts_cur[i])))
                else:
                    sinc_dvr_all_at_cur_grid.append(1.)  
            return sinc_dvr_all_at_cur_grid
        ##--------------------------------------##
        elif cur_dvr_type=="HO":
            ##-- constructing FBR==>DVR transformation matrix --##
            fbr_2_dvr=np.zeros((N_prim,N_prim)).tolist()
            hermite_index=np.zeros(N_prim).tolist()
            a,b=hermite.hermgauss(N_prim)
            a1,b1=a.tolist(),b.tolist()
            for i in range(len(fbr_2_dvr)):
                for j in range(len(fbr_2_dvr[i])):
                    hermite_index[j]+=1
                    fbr_2_dvr[i][j]=(math.sqrt(b1[i]))*(hermite.hermval(a1[i],hermite_index))*(1/((math.pi)**0.25))*(1/(math.sqrt((2**j)*math.factorial(j))))
                    hermite_index[j]-=1
            ##---------------------------------------------------##
            Q_shift_cur=self.Q_shift_arr[mode_index_id]          ## extracting the shift for DVR calculation for the current coordinate ##
            N_HO=len(self.sodvr_2_anharm_eigvecs[mode_index_id][0])    ## Number of HO functions needed ## 
            cur_coord_wo_shift=cur_coord_val-Q_shift_cur    ## value of HO wavefunction to be evaluated when there's no shift ##
            HO_func_all_nparr=np.array(self.prim_basis_val(N_HO,cur_coord_wo_shift))    
            HO_dvr_all_at_cur_grid=[]                       ## array for storing values of all HO DVR function values at the current grid ## 
            for i in range(len(fbr_2_dvr)):
                cur_coeff_nparr=np.array(fbr_2_dvr[i])        ## numpy coeff array for current DVR function ##
                cur_sodvr_val_at_cur_grid_tmp=(cur_coeff_nparr*HO_func_all_nparr).tolist()
                cur_sodvr_val_at_cur_grid=np.sum(cur_sodvr_val_at_cur_grid_tmp)     ## i-th SODVR function at the current grid ##
                HO_dvr_all_at_cur_grid.append(cur_sodvr_val_at_cur_grid)
            return HO_dvr_all_at_cur_grid
        ##--------------------------------------##
    def pobasis_val(self,mode_index,cur_coord_val,cur_coord_pobasis_index):
        """
        function for returning value of an one-dimensional podvr basis function at one-dim value @ multidim grid point
        cur_coord_pobasis_index (zero-based indexing) .. gives 1-d pofunction index whose value at the current grid is to be evaluated 
        it'll need transformation matrices for sodvr=> anharm_eigvecs => podvr basis
        """
        mode_index_id=self.mode_index_arr.index(mode_index)     ## python index for the concerned mode ##
        ### getting 1-d eigenvector matrices for dvr=>anharm_eigvecs & anharm_eigvecs => podvr functions ###
        dvr_2_anharm_eigvecs_cur,anharm_2_podvr_cur=np.array(self.sodvr_2_anharm_eigvecs[mode_index_id]),self.anharm_2_podvr_eigvecs[mode_index_id]
        anharm_2_podvr_cur_po=anharm_2_podvr_cur[cur_coord_pobasis_index]         ## array of coeffs for the current pobasis for the relevant coordinate ## 
        ###===============================================================##
        """ 
        for each po function all converged anharmonic eigenstates are to be evaluated at the current coordinate value 
        """
        dvr_func_all_at_cur_grid=np.array(self.dvr_basis_val_arr(mode_index,cur_coord_val))       ## calling dvr_basis_val_arr to get values of all dvr functions for the relevant coord at the current grid .. length equal to N_prim for the current coordinate ##
        anharm_all_at_cur_grid=[]                                   ## to store values of all converged anharm_eigvecs for the current coordinate ##
        for i in range(len(anharm_2_podvr_cur)):
            cur_anharm_eigvec_at_cur_coord_tmp=(dvr_2_anharm_eigvecs_cur[i]*dvr_func_all_at_cur_grid).tolist()
            cur_anharm_eigvec_at_cur_coord=np.sum(cur_anharm_eigvec_at_cur_coord_tmp)           ## i-th anharm-eigenfunction value at the current grid ##
            anharm_all_at_cur_grid.append(cur_anharm_eigvec_at_cur_coord)
        ##--------------------------------------------------------------------------##
        anharm_all_at_cur_grid_nparr=np.array(anharm_all_at_cur_grid)       ## corresponding numpy array for anharmonic eigenfunction ##
        """ length of anharm_all_at_cur_grid .. equal to no of converged eigenstates for the current coordinate .. no of 1-d pobasis for the current coordinate """
        cur_pobasis_val_at_cur_coord_tmp=(anharm_2_podvr_cur_po*anharm_all_at_cur_grid_nparr).tolist()
        cur_pobasis_val_at_cur_coord=np.sum(cur_pobasis_val_at_cur_coord_tmp)
        #--------returning pobasis value for current coordinate at the current grid --------##
        return cur_pobasis_val_at_cur_coord
    ##----------------------------------------------------------------------------------------##
    def get_podvr_func_val_at_multigrid(self,multigrid):
        """
        multigrid has normal mode values for various dimensions ..@ the multidim grid point
        dimension of multigrid equal to N_dim .. dimension for the eigenstate calculation
        """
        one_dim_index_arr=self.get_1_dim_index_arr()
        dpb_val_arr=[]                              ## array for saving dpb values for all basis at the current multigrid .. length equal to size of DPB ##
        for i in range(len(one_dim_index_arr)):
            dpb_val_cur=1.
            for j in range(len(one_dim_index_arr[i])):
                x=self.pobasis_val(self.mode_index_arr[j],multigrid[j],one_dim_index_arr[i][j])
                dpb_val_cur*=x
            dpb_val_arr.append(dpb_val_cur)
        return dpb_val_arr
    #----------------------------------------------------#
    def generate_meshgrid(self,other_coord_val_arr,plt_coord_1,plt_coord_2):
        """
         function for generating two dimensional meshgrid for plotting ..
         also generating multigrid for eigenfunction calculation at the 2-d 
         plotting grid as coordinates other than 2 plotting coords can be nonzero
         length of `other_coord_val_arr` is equal to N_dim .. dimension taken for eigenstate calculation 
         values of plotting coords will be kept at zero
        """
        delta_1,delta_2=(self.plt_q_1st_max-self.plt_q_1st_min)/(self.plt_q_1st_N_pts-1),(self.plt_q_2nd_max-self.plt_q_2nd_min)/(self.plt_q_2nd_N_pts-1)
        plt_coord_1_grid=[self.plt_q_1st_min+k*delta_1 for k in range(self.plt_q_1st_N_pts)]
        plt_coord_2_grid=[self.plt_q_2nd_min+k*delta_2 for k in range(self.plt_q_2nd_N_pts)]
        ##--------------------------------------------------##
        plt_grid_2d=[]
        for i in range(len(plt_coord_1_grid)):
            for j in range(len(plt_coord_2_grid)):
                plt_grid_2d.append([plt_coord_1_grid[i],plt_coord_2_grid[j]])
        ###------------------------------------------------###
        plt_coord_1_id,plt_coord_2_id=self.mode_index_arr.index(plt_coord_1),self.mode_index_arr.index(plt_coord_2)     ## getting python indices for plotting coordinates ##
        multigrid_4_eigstate_caln=[copy.deepcopy(other_coord_val_arr) for i in range(len(plt_grid_2d))]
        for i in range(len(multigrid_4_eigstate_caln)):
            multigrid_4_eigstate_caln[i][plt_coord_1_id]=plt_grid_2d[i][0]
            multigrid_4_eigstate_caln[i][plt_coord_2_id]=plt_grid_2d[i][1]
        ###------------------------------------------------###
        """ returning X,Y meshgrid as well as multigrid_4_eigstate_caln """
        return plt_coord_1_grid,plt_coord_2_grid,multigrid_4_eigstate_caln
    #----------------------------------------------------#
    def multidim_wavefunc_vals(self,other_coord_val_arr,plt_coord_1,plt_coord_2,multidim_eigvec_arr):
        """
        `multidim_eigvec_arr' is eigenvector coefficient array for a particular (e.g. I-th) eigenvector of the multidim. calculation
        """
        multigrid_4_eigfunc_caln=self.generate_meshgrid(other_coord_val_arr,plt_coord_1,plt_coord_2)[2]                    ## calling generate_meshgrid for N_dim grid for eigenstate calculation ##
        multidim_eigvec_nparr=np.array(multidim_eigvec_arr)     ## corresponding numpy array for the eigenvector coeff. array to be used later ##
        ###-------------------------------------------##
        psi_multi_vals_cur_eigstate=[]                  ## storing values of multidim. eigenstate at all points of the multidim. grid ##
        for i in range(len(multigrid_4_eigfunc_caln)):
            cur_dpb_val_nparr=np.array(self.get_podvr_func_val_at_multigrid(multigrid_4_eigfunc_caln[i]))       ## calling get_podvr_func_val_at_multigrid .. getting all DPB basis val @ current grid ##
            psi_multi_cur_grid_1=(multidim_eigvec_nparr*cur_dpb_val_nparr).tolist()
            psi_multi_cur_grid=np.sum(psi_multi_cur_grid_1)
            psi_multi_vals_cur_eigstate.append(psi_multi_cur_grid)
        ##-------------------------------------------------------##
        return psi_multi_vals_cur_eigstate


#####################################################################################


class multidim_eigenstate_vals(INDEX_ONE_2_MULTIDIM_DPB_BACK_AND_FORTH):
    """
    """
    def __init__(self,basis_size_arr,DPB_size,onedim_index_arr,pobasis_val_all_dim_arr,onedim_grid_all_coord_arr,meshgrid):
        super().__init__(basis_size_arr)
        self.multidim_index=DPB_size                            ## size of direct product basis ##
        self.onedim_index_arr=onedim_index_arr
        self.onedim_grid_all_coord_arr=onedim_grid_all_coord_arr
        self.pobasis_val_all_dim_arr=pobasis_val_all_dim_arr
        self.meshgrid=meshgrid
    def get_podvr_func_val_at_multigrid(self,multigrid):
        multigrid_id_arr=[]     ## index (zero-based) array for location of the grid for individual coordinates ##
        for i in range(len(multigrid)):
            cur_id=self.onedim_grid_all_coord_arr[i].index(multigrid[i])
            multigrid_id_arr.append(cur_id)
        print(multigrid_id_arr)
       # --------------------------------------- #
        dpb_val_arr=[]
        for i in range(len(self.onedim_index_arr)):
            dpb_val_cur=1.
            for j in range(len(self.onedim_index_arr[i])):
                x=self.pobasis_val_all_dim_arr[j][multigrid_id_arr[j]][self.onedim_index_arr[i][j]]
                dpb_val_cur*=x
            dpb_val_arr.append(dpb_val_arr)
        return dpb_val_arr
    # ------------------------------------ #
    def eigstate_multidim_vals(self,multidim_eigvec_arr):
        """
        """
        multidim_eigvec_nparr=np.array(multidim_eigvec_arr)
        cur_eigstate_vals=[]                ## storing values of multidim. eigenstate at all configurations of the meshgrid ##
        for i in range(len(self.meshgrid)):
            cur_dpb_val_nparr=np.array(self.get_podvr_func_val_at_multigrid(self.meshgrid[i]))
            psi_multi_cur_grid_1=(multidim_eigvec_nparr*cur_dpb_val_nparr).tolist()
            psi_multi_cur_grid=np.sum(psi_multi_cur_grid_1)
            print(psi_multi_cur_grid)
            cur_eigstate_vals.append(psi_multi_cur_grid)
        return cur_eigstate_vals
            


#######################################################################################

"""
PROGRAM FOR CALCULATING OVERLAP OF A PARTICULAR EIGENSTATE OF THE FULL HAMILTONIAN WITH DIRECT PRODUCT STATES GENERATED THROUGH 1-D EIGENSTATES OF ANHARMONIC CUTS
"""
class overlap(INDEX_ONE_2_MULTIDIM_DPB_BACK_AND_FORTH):
    """
    The loop has to run over all coeffs of the I-th( multidim. eigstate) array ..
    for each element in the array; corresponding index numbers for 1-d basis functions will be needed 
    Inherits attributes of class INDEX_ONE_2_MULTIDIM_DPB_BACK_AND_FORTH
    """
    def __init__(self,basis_size_arr,mode_index_arr,multidim_quanta_arr,multidim_eigvec_arr,anharm_2_podvr_all):
        super().__init__(basis_size_arr)                ## using `super` to inherit variables defined in __init__ in INDEX_ONE_2_MULTIDIM_DPB_BACK_AND_FORTH ## 
        """
        array for different quanta of excitation along different 1-d states of anharmonic oscillators .. array size equal to no. of dimensions considered in the calculation 
        indexing is usual indexing .. (starting from 1) not python indexing 
        """
        self.multiquanta_extn_arr=multidim_quanta_arr
        self.eigvec_cur=multidim_eigvec_arr             ## eigenvector coefficient array for a particluar (e.g. I-th) eigenvector of the multidim calculation ##
        self.anh_2_po_all_arr=anharm_2_podvr_all        ## multidimensional array of arrays containing anh=>po transformation matrices for all coords considered for calculation ##
        self.DPB_size=len(multidim_eigvec_arr)          ## Direct Product Basis size considered in the calculation ##
        self.mode_index_arr=mode_index_arr              ## Containing modes ( TS mode index ) considered for the multidim eigenstate calculation ##
    def get_1_dim_index_arr(self):
        """
        self.DPB_size is equal to length of direct product grid ..i.e. size of the direct product basis
        """
        onedim_index_arr=[]
        for i in range(1,self.DPB_size+1):
            cur_onedim_index=super().one_dim_indices(i) 
            onedim_index_arr.append(cur_onedim_index)
        ##-- returns one-dimensional index array for all basis in the direct product basis -- ##
        return onedim_index_arr
    """
    function for calculating 1-dim overlap integrals 
    """
    def get_1d_overlap(self,mode_index,pobasis_index,anharm_basis_index):
        """
        calculates overlap integral for a partiular coordinate ..
        mode_index gives a particular mode index
        pobasis_index gives a particular podvr basis index for the relevant coordinate (zero based index)
        anharm_basis_index gives 1-d anharm eigfunc index taken for calculate overlap .. usual index (one based index not python index)
        """
        mode_cur_id=self.mode_index_arr.index(mode_index)      
        anh_2_po_cur_arr=self.anh_2_po_all_arr[mode_cur_id]     ## Getting the relevant transformation matrix for the concerned mode ##
        coeff=anh_2_po_cur_arr[pobasis_index][anharm_basis_index-1] 
        return coeff
    def multidim_overlap(self):
        """
        Getting full overlap for a I-th multidim. eigenstate with a certain prepared direct product state 
        """
        onedim_index_arr=self.get_1_dim_index_arr()     ## getting 1-dim indices for all the elements in the multidim. eigenvector (zero based indexing) ##
        ### multidim eigenstate calculated in terms podvr basis .. each element of onedim_index_arr gives python indices for a particular element in the multidim eigenvector ## 
        TOT_overlap=0.
        for i in range(len(onedim_index_arr)):
            product_1d_overlap=1
            for j in range(len(onedim_index_arr[i])):
                mode_index_cur=self.mode_index_arr[j]                   ## TS mode index ##
                pobasis_index_cur=onedim_index_arr[i][j]                ## pobasis index (zero based index) ## 
                anharmbasis_index_cur=self.multiquanta_extn_arr[j]      ## anharm basis index (one based index) ##
                cur_1d_overlap=self.get_1d_overlap(mode_index_cur,pobasis_index_cur,anharmbasis_index_cur)
                product_1d_overlap*=cur_1d_overlap
            TOT_overlap+=self.eigvec_cur[i]*product_1d_overlap          ## adding contribution from each term in the multidim coeff array ##
        ###--- Returning overlap value--- ###
        return TOT_overlap

####################################################################################################################################################################
####################################################################################################################################################################
####################################################################################################################################################################

"""
routines for generating multidim quanta array up to 7 dimensions 
upto 3-dimensions it would be generated using direct product structure 
from 4- to 7 dimensions multidimensional quanta array will be generated 
using energy cut-off criteria 
"""

######------------------

#### FUNCTION FOR GENERATING MULTIDIMENSIONAL QUANTA ARRAY UP TO 3 DIMENSIONS 


def generate_dp_multidim_quanta_arr(max_quanta_arr):
    """
    max_quanta_arr contains maximum quantum no for a particular coordinate .. one-based indexing
    dimension of max_quanta_arr is two or three first element of max_quanta_arr is quanta for Q1
    genrating multidim_quanta_arr using direct product structure without any energy constraint on any mode
    """
    multidim_quanta_arr=[]
    max_quanta_arr_len=len(max_quanta_arr)
    if max_quanta_arr_len==2:
        for i in range(1,max_quanta_arr[0]+1,2):
            for j in range(1,max_quanta_arr[1]+1):
                multidim_quanta_arr.append([i,j])
                multidim_quanta_arr.append([i+1,j])
    elif max_quanta_arr_len==3:
        for i in range(1,max_quanta_arr[0]+1,2):
            for j in range(1,max_quanta_arr[1]+1):
                for k in range(1,max_quanta_arr[2]+1):
                    multidim_quanta_arr.append([i,j,k])
                    multidim_quanta_arr.append([i+1,j,k])
    return multidim_quanta_arr

######--------------------

#### ROUTINE FOR GENERATING MULTIDIMENSIONAL QUANTA ARRAY FOR 4- AND HIGHER DIMENSIONS

class GENERATE_MULTIDIM_QUANTA_ARR(INDEX_ONE_2_MULTIDIM_DPB_BACK_AND_FORTH):
    """
    This will generate multidimensional quanta array starting from a certain upper bound in 4- and higher dimensions
    """
    def __init__(self,basis_size_arr,E_tot_thr,multidim_quanta_arr_ub,eigvals_onedim_arr):
        """
        basis_size_arr to be given as input without the final i.e. highest frequency quanta mode 
        """
        super().__init__(basis_size_arr)                ## using `super` to inherit variables defined in __init__ in INDEX_ONE_2_MULTIDIM_DPB_BACK_AND_FORTH ## 
        """
        E_tot_thr is the total energy threshold
        multidim_quanta_arr_ub is initial quanta array (upper bound)
        eigvals_onedim_arr is array of arrays containing converged eigenvalues 
        for all converged states for all modes 
        """
        ## multidimensional quanta array indicated some i-th eigenstate for a particular mode ( not excitation) throughout ##
        self.E_tot_thr=E_tot_thr
        self.multidim_quanta_arr_ub=multidim_quanta_arr_ub      ## 1-based indexing ## 
        self.eigvals_onedim_arr=eigvals_onedim_arr
    def get_1_dim_index(self,multidim_index):
        """
        multidim_index is the multidimensional index (indexing start from 1)
        """
        cur_onedim_index=super().one_dim_indices(multidim_index)
        ## returns indices for individual modes (zero-based) corresponding to multidimensional index ##
        return cur_onedim_index
    def E_tot_min(self,i,multidim_quanta_arr_cur):
        E_tot_min=0.
        for j in range(len(self.eigevals_onedim_arr)):
            if j==i:
                E_tot_min+=self.eigvals_onedim_arr[j][multidim_quanta_arr_cur[j]-1]
            else:
                E_tot_min+=self.eigvals_onedim_arr[j][0]
        return E_tot_min
    ##------------------------------------------------------------------##
    def Tot_energy_calculator(self,multidim_cur_quanta_arr):
        """
        function for calculating total energy for a given quanta distribution
        multidim_cur_quanta_arr has 1-based indexing 
        """
        E_tot_cur=0.
        for i in range(len(multidim_cur_quanta_arr)):
            E_tot_cur+=self.eigvals_onedim_arr[i][multidim_quanta_arr[i]-1]
        return E_tot_cur
    ##------------------------------------------------------------------##
    def generate_multidim_quanta_arr_init(self):
        """
        starting multidimensional upper bound quanta array 
        generates minimum quanta for each mode that can be worked with
        """
        multidim_quanta_arr_cur=copy.deepcopy(self.multidim_quanta_arr_ub)
        N_count=len(multidim_quanta_arr_cur)
        N_count_arr=np.repeat(1,len(multidim_quanta_arr_cur))
        cur_ind=len(multidim_quanta_cur_arr)
        multidim_quanta_arr_init=[]
        while N_count!=0:
            cur_ind-=1
            while N_count_arr[cur_ind]!=0:
                tot_energy_cur=self.E_tot_min(cur_ind,multidim_quanta_arr_cur)
                if tot_energy_cur<self.E_tot_thr:
                    multidim_quanta_arr_init.append(multidim_quanta_arr_cur[cur_ind])
                    N_count_arr[cur_ind]=0
                    N_count-=1
                else:
                    multidim_quanta_arr_cur[cur_ind]-=1
        multidim_quanta_arr_init.reverse()
        return multidim_quanta_arr_init
    def multidim_quanta_arr_from_workable_start(self):
        """
        starting from workable initial quanta array 
        final multidimensional quanta array will be generated 
        """
        multidim_f_quanta_all_arr=[]                                                    ## final array for storing all multidimensional quanta ##
        multidim_quanta_arr_init=self.generate_multidim_quanta_arr_init()               ## getting initial workable array(1-based indexing) from function generate_multidim_quanta_arr_init ##
        N_tot_init=np.prod(multidim_quanta_arr_init)                                    ## total no of possibilities if direct product structure were to be used ##
        N_tot_init_wo_max_freq_mode = int(N_tot_init/multidim_quanta_arr_init[len(multidim_quanta_arr_init)-1])       ## total no of possibilities among all the modes excluding highest frequency mode ##
        N_count_cur=multidim_quanta_arr_init[len(multidim_quanta_arr_init)-1]                     
        while N_count_cur!=0:
            for i in range(1,N_tot_init_wo_max_freq_mode+1): 
                onedim_index_cur=self.get_1_dim_indeces_cur(i)    ## zero-based indexing .. without the highest frequency mode quanta ## 
                onedim_index_cur_np=np.array(onedim_index_cur)
                onedim_index_cur_list=(onedim_index_cur_np+1).tolist()
                onedim_index_cur_list.append(N_count_cur)
                E_tot_cur=self.Tot_energy_calculator(onedim_index_cur_list)
                if E_tot_cur>=self.E_tot_thr:
                    N_count_cur-=1
                    break
                else:
                    multidim_f_quanta_all_arr.append(onedim_index_cur_list)
        return multidim_f_quanta_all_arr
