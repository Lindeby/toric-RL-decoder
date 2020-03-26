from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.special import comb


##############################################################################################################
# load data 
##############################################################################################################

# bias data only four p rels 
data = pd.read_csv('results/results_mats/ground_state_list_bias.txt', sep=",", header=None)
npdata = np.array(data)

# full data set for prel 
data        = pd.read_csv('results/results_mats/ground_state_list_bias_all.txt', sep=",", header=None)
npdata_all  = np.array(data)

p_errors = np.linspace(0.05, 0.20, 16)


p_MWPM_5    = np.loadtxt('results/results_mats/MWPM_5.TXT')
p_MWPM_7    = np.loadtxt('results/results_mats/MWPM_7.TXT')
p_MWPM_9    = np.loadtxt('results/results_mats/MWPM_9.TXT')
lin_p_mwpm  = np.linspace(0.05, 0.20, 16)


p_MWPM_5_uncorr     = np.loadtxt('results/results_mats/uncorr_MWPM_5.TXT')
p_MWPM_7_uncorr     = np.loadtxt('results/results_mats/uncorr_MWPM_7.TXT')
p_MWPM_9_uncorr     = np.loadtxt('results/results_mats/uncorr_MWPM_9.TXT')
lin_p_mwpm_uncorr   = np.linspace(0.05, 0.19, 8)

p_RL_5      = np.loadtxt('results/results_mats/RL_5.txt')
p_RL_5_06   = np.loadtxt('results/results_mats/RL_5_06.txt')

p_RL_7          = np.loadtxt('results/results_mats/RL_7.txt')
p_RL_7_06       = np.loadtxt('results/results_mats/RL_7_06.txt')
p_RL_9          = np.loadtxt('results/results_mats/RL_9.txt')
p_RL_9_06       = np.loadtxt('results/results_mats/RL_9_06.txt')
lin_p_rl_full   = np.linspace(0.06, 0.20, 8)


p_RL_5_uncorr               = np.loadtxt('results/results_mats/only_x_RL_5.txt')
p_RL_5_uncorr_biased_plot   = np.loadtxt('results/results_mats/uncorr_MWPM_5_biased_plot.TXT')
p_RL_7_uncorr               = np.loadtxt('results/results_mats/only_x_RL_7.txt')
p_RL_9_uncorr               = np.loadtxt('results/results_mats/only_x_RL_9.txt')
lin_p_rl_uncorr             = np.linspace(0.05, 0.19, 8)
loglog_5                    = np.loadtxt('results/results_mats/loglog_5.txt')
loglog_7                    = np.loadtxt('results/results_mats/loglog_7.txt')


##############################################################################################################
# create plot 
##############################################################################################################


def create_surface_plot():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    prediction_list_p_error = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
    p_z_prob_list = [0.00000001,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


    Y = prediction_list_p_error
    X = p_z_prob_list


    X, Y = np.meshgrid(X, Y)
    #print(X.shape)
    #print(Y.shape)
    #print(data.shape)
    Z = npdata_all

    ax.set_xlabel('percentage of P_z error')
    ax.set_ylabel('overall error probability')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    
    # Customize the z axis.
    ax.set_zlim(0, 1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)


    # rotate the axes and update
    ax.view_init(35, 40) #rotate up down  ,rotate right left 


    #for angle in range(0, 360):
    #    ax.view_init(30, angle)
        #plt.draw()
        #plt.pause(.001)



    plt.savefig('plots/surface_plot.pdf')


def create_contour_plot():
    fig = plt.figure()
    ax = plt.subplot()

    Y = prediction_list_p_error
    X = p_z_prob_list


    X, Y = np.meshgrid(X, Y)
    #print(X.shape)
    #print(Y.shape)
    #print(data.shape)
    Z = data

    ax.set_xlabel('percentage of P_z error')
    ax.set_ylabel('overall error probability')

    # Plot the surface.
    surf = ax.contourf(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    
    # Customize the z axis.
    #ax.set_zlim(0, 1)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)


    # rotate the axes and update
    #ax.view_init(35, 40) #rotate up down  ,rotate right left 


    #for angle in range(0, 360):
    #    ax.view_init(30, angle)
        #plt.draw()
        #plt.pause(.001)



    plt.savefig('plots/graph.pdf')


def uncorrelated_noise():

    fig, ax = plt.subplots(1,1,figsize=(8,6))
    #plt.title('uncorrelated noise')
    plt.ylabel(r'$P_s$')
    plt.xlabel('$p$')

    ax.plot(lin_p_rl_uncorr, p_RL_5_uncorr, 'o', color = 'C0', label='RL (d=5)')
    ax.plot(lin_p_rl_uncorr, p_RL_7_uncorr, 's', color = 'C1', label='RL (d=7)')
    ax.plot(lin_p_rl_uncorr, p_RL_9_uncorr, '^', color = 'C2', label='RL (d=9)')
    ax.plot(lin_p_mwpm_uncorr, p_MWPM_5_uncorr, '--', color = 'C0', label='MWPM (d=5)')
    ax.plot(lin_p_mwpm_uncorr, p_MWPM_7_uncorr, ':', color = 'C1', label='MWPM (d=7)')
    ax.plot(lin_p_mwpm_uncorr, p_MWPM_9_uncorr, '--', color = 'C2', label='MWPM (d=9)')

    xticks = [0.05, 0.1, 0.15, 0.2]
    ax.set_xticks(xticks)
    
    yticks = [0.4, 0.6, 0.8, 1]
    ax.set_yticks(yticks)

    fig.tight_layout()

    plt.legend()
    plt.savefig('plots/uncorrelated_noise.pdf')


def depolarized_noise():

    fig, ax = plt.subplots(1,1,figsize=(8,6))
    plt.ylabel(r'$P_s$')
    plt.xlabel('$p$')
    
    ax.plot(np.linspace(0.05, 0.19, 8), p_RL_5, 'o', color = 'C0', label='RL (d=5)')
    ax.plot(np.linspace(0.06, 0.20, 8), p_RL_5_06, 'o', color = 'C0')

    ax.plot(np.linspace(0.05, 0.19, 8), p_RL_7, 's', color = 'C1', label='RL (d=7)')
    ax.plot(np.linspace(0.06, 0.20, 8), p_RL_7_06, 's', color = 'C1')
    
    ax.plot(np.linspace(0.05, 0.19, 8), p_RL_9, '^', color = 'C2', label='RL (d=9)')
    ax.plot(np.linspace(0.06, 0.20, 8), p_RL_9_06, '^', color = 'C2')
    
    ax.plot(lin_p_mwpm, p_MWPM_5, color = 'C0', label='MWPM (d=5)')
    ax.plot(lin_p_mwpm, p_MWPM_7, ':', color = 'C1', label='MWPM (d=7)')
    ax.plot(lin_p_mwpm, p_MWPM_9, '--', color = 'C2', label='MWPM (d=9)')

    # print(p_RL_5)
    # print(lin_p_rl_full)
    xticks = [0.05, 0.1, 0.15, 0.2]
    ax.set_xticks(xticks)
    
    yticks = [0.4, 0.6, 0.8, 1]
    ax.set_yticks(yticks)

    fig.tight_layout()

    plt.legend()
    plt.title('Depolarized Noise')
    plt.show()
    # plt.savefig('plots/depolarized_noise.pdf')

    # plt.close()


def bias_syndrome_generation():
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    
    plt.ylabel('$P_s$')
    plt.xlabel('$p$')

    y = np.array([0.99425, 0.98488, 0.96784, 0.9437,  0.9112,  0.86804, 0.82557, 0.8])
    x = np.array([0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.095])
    


    y_new = y * y
    x_new = 2 * x

    y_add = np.array([0.995])
    y_new = np.append(y_add, y_new)

    #ax.plot(x_temp, y_temp, 'o-', color = 'C7', label='New data')
    ax.plot(x_new, y_new, '--', color = 'C8', label='$MWPM(p/2)^2$')

    # p rel 0 only x and y errors 
    data_temp = npdata[:,0]
    ax.plot(p_errors[::2], data_temp[::2], '^', color = 'C1', label='$p_{rel} = 0$')
    
    # depolarized noise p rel = 0.1
    data_temp = npdata_all[:,1]
    ax.plot(p_errors[::2], data_temp[::2], 'o', color = 'C4', label='$p_{rel} = 0.1$')

    # depolarized noise p rel = 0.3333
    data_temp = npdata[:,1]
    ax.plot(p_errors[::2], data_temp[::2], 'o', color = 'C0', label='$p_{rel} = 1/3$')
    #ax.plot(lin_p_rl_full, p_RL_5, 'o', color = 'C0', label='p_rel = 0.333 (depolarized noise)')
    
    # depolarized noise p rel = 0.7
    data_temp = npdata_all[:,7]
    ax.plot(p_errors[::2], data_temp[::2], 'o', color = 'C3', label='$p_{rel} = 0.7$')
    
    # uncorrelated noise p rel = 1
    ax.plot(lin_p_rl_uncorr, p_RL_5_uncorr, 's', color = 'C2', label='$p_{rel} = 1$')
    
    # benchmark MWPM uncorrelated noise
    ax.plot(lin_p_mwpm_uncorr, p_MWPM_5_uncorr,'--', color = 'C2', label='$MWPM(p)$')
    
    #x_temp = np.array([0.1, 0.14, 0.18])
    #y_temp = np.array([0.93654, 0.830486, 0.675059])
    xticks = [0.05, 0.1, 0.15, 0.2]
    ax.set_xticks(xticks)
    
    yticks = [0.4, 0.6, 0.8, 1]
    ax.set_yticks(yticks)

    fig.tight_layout()

    plt.legend()
    plt.savefig('plots/biased_syndrome_generation.pdf')
    plt.close()



def loglog_plot():

    #plt.title('low p error rate')
    
    def asymptotics_rl(p, d):
        k = np.ceil(d/2)
        # return 4 *d * ( comb(d,k) + k * comb(d, k-1)) / (comb(2*d**2, k) * k**3)
        # return 4*d * ((comb(d, k)) * (p/3)**k + k*comb(d, k -1)*(2*p/3)**(k-1))
        # return 4 * d * (comb(d,k) * (p/3)**k + k * (p/3) * comb(d, k-1) * ((2/3)*p)**(k-1))
        return 4 * d * (p/3)**k * (comb(d,k) + d  * comb(d-1, k-1))


    def asymptotics_MWPM(p, d):
        k = int(d/2) + 1
        number_of_ways = 0 
        for n_y in range(k):
            #number_of_ways += comb(k,n_y) * comb(d, k-n_y) * comb(d-k+n_y, n_y)
            number_of_ways += comb(d, n_y) * comb(d-n_y, k - n_y)
        # probability logical error MWPM 
        p_l_mwpm = 2 * 2 * d * number_of_ways * ((1/3)*p)**k
        return p_l_mwpm
        

    p_errors = loglog_5[:,0]

    p_l_asymptotics5 = asymptotics_rl(p_errors, 5)
    p_l_asymptotics_MWPM5 = asymptotics_MWPM(p_errors, 5)
    #print('p_l_asymptotics5:', p_l_asymptotics5)
    #print('p_l_asymptotics_MWPM5:', p_l_asymptotics_MWPM5)

    p_l_asymptotics7 = asymptotics_rl(p_errors, 7)
    p_l_asymptotics_MWPM7 = asymptotics_MWPM(p_errors, 7)

    fig, ax = plt.subplots(1,1,figsize=(8,6))
    # plt.title('Asymptotic behaviour for d=5 and d=7')
    plt.ylabel(r'$P_L$')
    plt.xlabel('$p$')
    ax.loglog(loglog_5[:,0], loglog_5[:,1], 'o', color = 'C0', label='RL, d=5')
    ax.loglog(p_errors, p_l_asymptotics5, '--', color = 'C0', label='MCC, d=5')
    ax.loglog(p_errors, p_l_asymptotics_MWPM5, ':', color = 'C0', label='MWPM, d=5')


    ax.loglog(loglog_7[:,0], loglog_7[:,1], '^', color = 'C1', label='RL, d=7')
    ax.loglog(p_errors, p_l_asymptotics7,  '--', color = 'C1', label='MCC, d=7')
    ax.loglog(p_errors, p_l_asymptotics_MWPM7, ':', color = 'C1', label='MWPM, d=7')
    
    #xticks = [0.05, 0.1, 0.15, 0.19]
    #ax.set_xticks(xticks)
    
    yticks = [10e-17, 10e-13, 10e-9, 10e-5]
    ax.set_yticks(yticks)

    fig.tight_layout()


    
    ax.legend()
    plt.savefig('plots/loglog.pdf')
    plt.close()

#create_surface_plot()
#create_contour_plot()

# uncorrelated_noise()
depolarized_noise()
# bias_syndrome_generation()
# loglog_plot()
