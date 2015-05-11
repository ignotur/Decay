from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import *

#######################################################
## I reproduce here the numerical method 
## developped in Marchant et al (2014) ArXiv:1410.5833
## to study the magnetic field decay in the 
## crust of neutron stars
#######################################################

## This function describes the distribution of density and 
## conductivity along the crust according to the article
## by Gourgouliatos and Cumming (2013) ArXiv: 1311.7004

def crust_model (r, theta):
	r_star = 1.0e6
	n_e = 2.5e34*np.power((1.0463*r_star - r)/(0.0463*r_star), 4.0)  ## 2.5e36 at the base of the crust
	sigma = 2.0*np.power(n_e, 0.66666667)
	
	return [n_e, sigma]

## We introduce here the uniform grid in angles comparatively
## to uniform in \cos\theta grid in Gorgouliatos and Cumming

def crust_model_grid (n, m):
	r_star = 1.0e6
	r_grid  = np.linspace (0.9*r_star, 1.0*r_star, num = n)
	mu_grid = np.linspace (-pi/2.0, pi/2.0, num = m) 

	ne = np.zeros((n, m))
	sigma = np.zeros((n, m))

	for j in range (0, n):
		for k in range (0, m):
			ne[j][k], sigma[j][k] = crust_model (r_grid[j], mu_grid[k])
	
	return [r_grid, mu_grid, ne, sigma]

## Here we generate the initial field configuration.
## We start from pure poloidal field

def initial_pure_poloidal (n, m, B):
	r_star = 1.0e6
	r_grid  = np.linspace (0.9*r_star, 1.0*r_star, num = n)
	mu_grid = np.linspace (-pi/2.0, pi/2.0, num = m) 

	phi = np.zeros((n,m))
	I   = np.zeros((n,m))  
	for j in range (2, n):
		for k in range (0, m):
		 	phi  [j][k] = 5e29 * sqrt(1 - np.power(mu_grid[k], 2.0)) /r_grid[j]  ## This generates another component
			I [j][k] = 0
	return [r_grid, mu_grid, phi, I]

## We use this function to plot the distribution of poloidal and toroidal components
## in the crust of neutron star

def plot_res (n, m, phi, I, r_grid, mu_grid):
	X = np.zeros((n, m))
	Y = np.zeros((n, m))
	X1 = np.zeros((n, m))
	Y1 = np.zeros((n, m))



	for i in range (0, n):
		for j in range (0, m):
			Y[i][j]  = r_grid[i] * cos(mu_grid[j])
			X[i][j]  = r_grid[i] * sin(mu_grid[j])
			Y1[i][j] = r_grid[i] * cos(mu_grid[j])
			X1[i][j] =-r_grid[i] * sin(mu_grid[j])



	plt.pcolormesh(X,Y,phi)
	plt.pcolormesh(X1, Y1, I)
	plt.show()

## This is the basic function which computes the value of the toroidal
## and the poloidal component at the next time step

def one_time_step (n, m, phi, I, r_grid, mu_grid):
	e_charge = 4.80320451e-10
	c = 3.0e10
	c2 = pow(c, 2.0)
	r_star = 1.0e6 
	h =1.0e7 ## Should be analysed again (!!!)
	delta_r  = 0.1 * r_star / float(n)
	delta_mu = pi / float(m)
	delta_t  = 0.25 * h * delta_r * delta_mu 
	const1 = c / (4.0 * 3.1415926 * e_charge)     ## c / (4 * pi * e)

        r_grid, mu_grid, ne, sigma = crust_model_grid (n, m)

	phi_new = np.zeros((n,m))
	I_new = np.zeros((n,m))

	for j in range (1, n-1):                 ## r  change 
		for k in range (1, m-1):         ## mu change 
			r_b = c*1.0e14/(4.0*pi*ne[j][k]*e_charge)*sigma[j][k]			## temporal equation eq. (3) from ArXiv 1410.5833
			r_b_inv = 1.0/r_b
			hi = const1 / (ne[j][k] * pow(r_grid[j] * sin(mu_grid[k]), 2.0)) 
			shafranov_phi = (phi[j+1][k] + phi[j-1][k] - 2.0 * phi[j][k]) / pow(delta_r, 2.0) + (phi[j][k+1] + phi[j][k-1] - 2.0*phi[j][k]) / pow(r_grid[j], 2.0) / pow(mu_grid[k], 2.0) - 1.0/tan(mu_grid[k]) * (phi[j][k+1] - phi[j][k-1])/2.0/pow(r_grid[j], 2.0) / delta_mu[k]

			delta_phi_delta_t = sin(mu_grid[k]) * hi / (4.0*delta_r*delta_mu) * ( (I[j][k+1] - I[j][k-1])*(phi[j+1][k] - phi[j-1][k])  - (I[j+1][k] - I[j-1][k])*(phi[j][k+1] - phi[j][k-1])) + r_b_inv * shafranov_phi	

			F_r     = -hi * shafranov_phi * (phi[j][k+1] - phi[j][k-1]) / delta_mu - hi * I * (I[j][k+1] - I[j][k-1]) / delta_mu - r_b_inv / (sigma[j][k] * sin(mu_grid[k]) * (I[j+1][k] - I[j-1][k])/delta_r
			F_theta =  hi * shafranov_phi * phi[j][k] * (phi[j+1][k]-phi[j-1][k]) / delta_r + hi * I[j][k] * (I[j+1][k]-I[j-1][k]) / delta_r - r_b_inv / (sigma[j][k] * pow(r_grid[j], 2.0) * sin(mu_grid[k])) * (I[j][k+1]-I[j][k-1])/delta_mu

			delta_I_delta_t = - sin(mu_grid[k])/delta_r * (F_r_pr - F_r_lr) - sin(mu_grid[k])/delta_mu * (F_theta_rmu - F_theta_lmu)

