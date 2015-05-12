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
	mu_grid = np.linspace (-pi, 0, num = m) 

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
	mu_grid = np.linspace (-pi, 0.0, num = m) 

	phi = np.zeros((n,m))
	I   = np.zeros((n,m))  
	for j in range (1, n):
		for k in range (0, m):
		 	phi  [j][k] = -3e28 * sin(mu_grid[k]) /r_grid[j]  ## This generates another component (5e29)
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
	h =5.0e8 ## Should be analysed again (!!!)
	delta_r  = 0.1 * r_star / float(n)
	delta_mu = pi / float(m)
	delta_t  = 0.25 * h * delta_r * delta_mu 
	const1 = c / (4.0 * 3.1415926 * e_charge)     ## c / (4 * pi * e)

        r_grid, mu_grid, ne, sigma = crust_model_grid (n, m)

	phi_new = np.zeros((n,m))
	I_new = np.zeros((n,m))

#	print 'here is pi: ', pi

	for j in range (1, n-2):                 ## r  change 
		for k in range (2, m-2):         ## mu change
			eta = pow(c, 2.0) / (4*pi*sigma[j][k]) 
			r_b = c*1.0e14/(4.0*pi*ne[j][k]*e_charge*eta)			
			r_b_inv = 1.0/r_b
			hi = const1 / (ne[j][k] * pow(r_grid[j] * sin(mu_grid[k]), 2.0)) 

			## Differential operator of Grad-Shafranov
			shafranov_phi = (phi[j+1][k] + phi[j-1][k] - 2.0 * phi[j][k]) / pow(delta_r, 2.0) + (phi[j][k+1] + phi[j][k-1] - 2.0*phi[j][k]) / (pow(r_grid[j], 2.0) * pow(delta_mu, 2.0)) - 1.0/tan(mu_grid[k]) * (phi[j][k+1] - phi[j][k-1])/2.0/pow(r_grid[j], 2.0) / delta_mu

			delta_phi_delta_t = sin(mu_grid[k]) * hi / (4.0*delta_r*delta_mu) * ( (I[j][k+1] - I[j][k-1])*(phi[j+1][k] - phi[j-1][k])  - (I[j+1][k] - I[j-1][k])*(phi[j][k+1] - phi[j][k-1])) + r_b_inv * shafranov_phi * eta	

	#		F_r     = - hi * shafranov_phi * (phi[j][k+1] - phi[j][k-1]) / delta_mu - hi * I[j][k] * (I[j][k+1] - I[j][k-1]) / delta_mu - r_b_inv / (sigma[j][k] * sin(mu_grid[k])) * (I[j+1][k] - I[j-1][k])/delta_r

			##################################################################################
			## Prepare differential operators at upper and bottom level for F_r_pr and F_r_pl
			##################################################################################
			eta_rp = pow(c, 2.0) / (4*pi*0.5*(sigma[j+1][k]+sigma[j][k])) 
			eta_lp = pow(c, 2.0) / (4*pi*0.5*(sigma[j-1][k]+sigma[j][k])) 

			r_b = c*1.0e14/(4.0*pi*0.5*(ne[j+1][k]+ne[j][k])*e_charge*eta_rp)			
			r_b_inv_pr = 1.0/r_b

			r_b = c*1.0e14/(4.0*pi*0.5*(ne[j-1][k]+ne[j][k])*e_charge*eta_lp)			
			r_b_inv_lr = 1.0/r_b



			shafranov_phipr = (phi[j+2][k] + phi[j][k] - 2.0 * phi[j+1][k]) / pow(delta_r, 2.0) + (phi[j+1][k+1] + phi[j+1][k-1] - 2.0*phi[j+1][k]) / (pow(r_grid[j+1], 2.0) * pow(delta_mu, 2.0)) - 1.0/tan(mu_grid[k]) * (phi[j+1][k+1] - phi[j+1][k-1])/2.0/pow(r_grid[j], 2.0) / delta_mu


			shafranov_philr = (phi[j][k] + phi[j-2][k] - 2.0 * phi[j-1][k]) / pow(delta_r, 2.0) + (phi[j-1][k+1] + phi[j-1][k-1] - 2.0*phi[j-1][k]) / (pow(r_grid[j-1], 2.0) * pow(delta_mu, 2.0)) - 1.0/tan(mu_grid[k]) * (phi[j-1][k+1] - phi[j-1][k-1])/2.0/pow(r_grid[j-1], 2.0) / delta_mu


			hipr = const1 / (crust_model(0.5*(r_grid[j+1] + r_grid[j]), 0.0)[0] * pow(0.5*(r_grid[j+1] + r_grid[j]) * sin(mu_grid[k]), 2.0)) 
			hilr = const1 / (crust_model(0.5*(r_grid[j-1] + r_grid[j]), 0.0)[0] * pow(0.5*(r_grid[j-1] + r_grid[j]) * sin(mu_grid[k]), 2.0)) 



			F_r_pr     = -hipr * shafranov_phipr * 0.5 * (phi[j+1][k+1] - phi[j+1][k-1]) / delta_mu - hipr * I[j+1][k] * 0.5*(I[j+1][k+1] - I[j+1][k-1]) / delta_mu - r_b_inv_pr * eta_rp / sin(mu_grid[k]) * (I[j+1][k] - I[j][k])/delta_r

			F_r_lr     = -hilr * shafranov_philr * 0.5 * (phi[j-1][k+1] - phi[j-1][k-1]) / delta_mu - hilr * I[j-1][k] * 0.5*(I[j-1][k+1] - I[j-1][k-1]) / delta_mu - r_b_inv_lr * eta_lp / sin(mu_grid[k]) * (I[j][k] - I[j-1][k])/delta_r

			########################################################################################
			## Prepare differential operators to the left and right for F_theta_pmu and F_theta_rmu
			########################################################################################
			eta_pmu = pow(c, 2.0) / (4*pi*0.5*(sigma[j][k+1]+sigma[j][k])) 
			eta_lmu = pow(c, 2.0) / (4*pi*0.5*(sigma[j][k-1]+sigma[j][k])) 

			r_b = c*1.0e14/(4.0*pi*0.5*(ne[j][k+1]+ne[j][k])*e_charge*eta_pmu)			## Reminder - use the correct value for B_0 (!!!)
			r_b_inv_pmu = 1.0/r_b

			r_b = c*1.0e14/(4.0*pi*0.5*(ne[j][k-1]+ne[j][k])*e_charge*eta_lmu)			## Reminder - use the correct value for B_0 (!!!)
			r_b_inv_lmu = 1.0/r_b


#			F_theta =  hi * shafranov_phi * phi[j][k] * (phi[j+1][k]-phi[j-1][k]) / delta_r + hi * I[j][k] * (I[j+1][k]-I[j-1][k]) / delta_r - r_b_inv / (sigma[j][k] * pow(r_grid[j], 2.0) * sin(mu_grid[k])) * (I[j][k+1]-I[j][k-1])/delta_mu

			hipmu = const1 / (ne[j][k+1] * pow(r_grid[j] * sin(mu_grid[k+1]), 2.0)) 
			hilmu = const1 / (ne[j][k-1] * pow(r_grid[j] * sin(mu_grid[k-1]), 2.0)) 

			shafranov_phi_pmu = (phi[j+1][k+1] + phi[j-1][k+1] - 2.0 * phi[j][k+1]) / pow(delta_r, 2.0) + (phi[j][k+2] + phi[j][k] - 2.0*phi[j][k+1]) / (pow(r_grid[j], 2.0) * pow(delta_mu, 2.0)) - 1.0/tan(mu_grid[k+1]) * (phi[j][k+2] - phi[j][k])/2.0/pow(r_grid[j], 2.0) / delta_mu


			shafranov_phi_lmu = (phi[j+1][k-1] + phi[j-1][k-1] - 2.0 * phi[j][k-1]) / pow(delta_r, 2.0) + (phi[j][k] + phi[j][k-2] - 2.0*phi[j][k-1]) / (pow(r_grid[j], 2.0) * pow(delta_mu, 2.0)) - 1.0/tan(mu_grid[k-1]) * (phi[j][k] - phi[j][k-2])/2.0/pow(r_grid[j], 2.0) / delta_mu


			F_theta =  hi * shafranov_phi * 0.5* (phi[j+1][k]-phi[j-1][k]) / delta_r + hi * I[j][k] * 0.5 * (I[j+1][k]-I[j-1][k]) / delta_r - r_b_inv * eta / (pow(r_grid[j], 2.0) * sin(mu_grid[k])) * 0.5 * (I[j][k+1]-I[j][k-1])/delta_mu


			F_theta_pmu =  hipmu * shafranov_phi_pmu * 0.5 * (phi[j+1][k+1]-phi[j-1][k+1]) / delta_r + hipmu * 0.5*(I[j][k+1] + I[j][k]) * 0.5 *(I[j+1][k+1]-I[j-1][k+1]) / delta_r - r_b_inv_pmu * eta_pmu / (pow(r_grid[j], 2.0) * sin(0.5*(mu_grid[k+1]+mu_grid[k]))) * (I[j][k+1]-I[j][k])/delta_mu

			F_theta_lmu =  hilmu * shafranov_phi_lmu * 0.5 * (phi[j+1][k-1]-phi[j-1][k-1]) / delta_r + hilmu * 0.5*(I[j][k-1] + I[j][k]) * 0.5 *(I[j+1][k-1]-I[j-1][k-1]) / delta_r - r_b_inv_lmu * eta_lmu / (pow(r_grid[j], 2.0) * sin(0.5*(mu_grid[k-1]+mu_grid[k]))) * (I[j][k]-I[j][k-1])/delta_mu


			delta_I_delta_t = - sin(mu_grid[k])/delta_r * (F_r_pr - F_r_lr) - sin(mu_grid[k])/delta_mu * (F_theta_pmu - F_theta_lmu)

			phi_new[j][k] = phi[j][k] + delta_t * delta_phi_delta_t
			I_new  [j][k] = I  [j][k] + delta_t * delta_I_delta_t


			## Important reminder (!!!) you should compute r_b_inv for upper and lower layer as well (!!!)

#			## Should correctly implement the boundary conditions (!!!)

	## Outer vacuum boundary condition

	l_polinom = 20
	al = []
	for l in range (1,l_polinom): 
		var = 1.0 / (l + 1.0) * sqrt(3.1415926*(2.0*l+1.0)) 
		res = 0.0
		for j in range (0, m-1):
			legendre_fun = lpmn(1, l_polinom, cos(mu_grid[j]))[0][1]
			res = res + 0.5 * (phi_new[n-3][j] + phi_new[n-3][j+1]) * legendre_fun[l] * delta_mu 
		var = res * var
		al.append(var)
	

	for k in range (1, m-1):

#		legendre_fun = lpmn(1, 20, z)[0][1]
	
		res = 0.0
		legendre_fun = lpmn(1, l_polinom, cos(mu_grid[k]))[0][1]
		for l in range (1,l_polinom):
			res = res - al[l-1] / pow(r_star, 1.0) * sqrt((2.0*l+1.0) / (4.0*3.1415926))  * legendre_fun[l] * sin(mu_grid[k])
		res = res * 2 * delta_r


		phi_new[n-1][k] = phi[n-3][k] + res
#		print phi_new[n-1][k]
		phi_new[n-2][k] = phi[n-4][k] + res 

		I_new[n-1][k] = 0.0
#		I_new[n-2][k] = 0.0
		I_new[0][k] = 0.0
		phi_new[0][k] = 0.0

	for j in range (0, n):
		phi_new[j][0] = 0.0
		phi_new[j][m-1] = 0.0
		I_new[j][0] = 0.0
		I_new[j][m-1] = 0.0


	return [phi_new, I_new, r_grid, mu_grid]

n = 30
m = 100

r_grid, mu_grid, phi, I = initial_pure_poloidal (n, m, 10)
plot_res (n, m, phi, I, r_grid, mu_grid)

for i in range (0, 1050):
	phi_new, I_new, r_grid, mu_grid = one_time_step (n, m, phi, I, r_grid, mu_grid)
	print i, np.max(I_new), np.min(I_new), np.max(phi_new), np.min(phi_new)
	if (i % 200 == 10):
		plot_res (n, m, phi_new, I_new, r_grid, mu_grid)
	phi = phi_new
	I = I_new
	

