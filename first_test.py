from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import *

def crust_model (r, theta):
	r_star = 1.0e6
	n_e = 2.5e34*np.power((1.0463*r_star - r)/(0.0463*r_star), 4.0)  ## 2.5e36 at the base of the crust
	sigma = 2.0*np.power(n_e, 0.66666667)
	
	return [n_e, sigma]

def crust_model_grid (n, m):
	r_star = 1.0e6
	r_grid  = np.linspace (0.9*r_star, 1.0*r_star, num = n)
	mu_grid = np.linspace (-1.0, 1.0, num = m) 
#	ne, sigma = crust_model(r_grid, mu_grid)
	ne = np.zeros((n, m))
	sigma = np.zeros((n, m))

	for j in range (0, n):
		for k in range (0, m):
			ne[j][k], sigma[j][k] = crust_model (r_grid[j], mu_grid[k])
			
	return [r_grid, mu_grid, ne, sigma]

def initial_pure_poloidal (n, m, B):
#	phi = 1.0 ## temporary should be related with B
#	I   = 0.0 ## temporary should be related with B
#	nabla_phi = [0, 0, 1.0/(r * sin(theta))]
	r_star = 1.0e6
	r_grid  = np.linspace (0.9*r_star, 1.0*r_star, num = n)
	mu_grid = np.linspace (-1.0, 1.0, num = m) 

	print mu_grid


	phi = np.zeros((n,m))
	I   = np.zeros((n,m))  

	for j in range (2, n):
		for k in range (0, m):
		 	phi  [j][k] = 1e20 * sqrt(1 - np.power(mu_grid[k], 2.0)) / r_grid[j]  ## This generates another component
			I [j][k] = 0
#			I  [j][k] = 1e16 * cos(3.0*acos(mu_grid[k])) / pow(r_grid[j], 1.0)
#			phi  [j][k] = 1e20 * sin(3.0*acos(mu_grid[k])) / pow(r_grid[j], 1.0)

	return [r_grid, mu_grid, phi, I]


def plot_res (n, m, phi, I, r_grid, mu_grid):
	X = np.zeros((n, m))
	Y = np.zeros((n, m))
	X1 = np.zeros((n, m))
	Y1 = np.zeros((n, m))



	for i in range (0, n):
		for j in range (0, m):
			Y[i][j] = r_grid[i] * mu_grid[j]
			X[i][j] = r_grid[i] * sqrt(1 - pow(mu_grid[j], 2.0))
			Y1[i][j] = r_grid[i] * mu_grid[j]
			X1[i][j] = -r_grid[i] * sqrt(1 - pow(mu_grid[j], 2.0))



	plt.pcolormesh(X,Y,phi)
	plt.pcolormesh(X1, Y1, I)
	plt.show()

def one_time_step (n, m, phi, I, r_grid, mu_grid):
	e_charge = 4.80320451e-10
	c = 3.0e10
	c2 = pow(c, 2.0)
	r_star = 1.0e6 
	h = 1e8 ## Should be analysed again (!!!)
	delta_r  = 0.1 * r_star / float(n)
	delta_mu = 2.0 / float(m)
	delta_t  = 0.25 * h * delta_r * delta_mu 
	const1 = c / (4.0 * 3.1415926 * e_charge)     ## c / (4 * pi * e)

        r_grid, mu_grid, ne, sigma = crust_model_grid (n, m)

	phi_new = np.zeros((n,m))
	I_new = np.zeros((n,m))

#        print delta_t / 3.2e7 

	for j in range (1, n-2):                 ## r  change 
		for k in range (1, m-1):         ## mu change 
			## for each point inside the grid we compute time derivative 
			## for scalar functions phi and I
			sintheta  = sqrt(1.0 - pow(mu_grid[k], 2.0))
			sintheta2 = sintheta  * sintheta
			sintheta3 = sintheta2 * sintheta
			sintheta4 = sintheta3 * sintheta
			costheta = mu_grid[k]
			r  = r_grid[j]
			r2 = pow(r, 2.0)
			r3 = pow(r, 3.0)
			r4 = pow(r, 4.0)

			phi_r   = (phi[j+1][k] - phi[j-1][k]) / (2.0 * delta_r)
			phi_mu  = (phi[j][k+1] - phi[j][k-1]) / (2.0 * delta_mu)

			phi_rr  = (phi[j-1][k] - 2.0 * phi[j][k] + phi[j+1][k]) / pow(delta_r, 2.0)
			phi_mumu= (phi[j][k-1] - 2.0 * phi[j][k] + phi[j][k+1]) / pow(delta_mu, 2.0)

			phi_rp  = (phi[j+1][k+1] - phi[j-1][k+1]) / (2.0 * delta_r)
			phi_rl  = (phi[j+1][k-1] - phi[j-1][k-1]) / (2.0 * delta_r)
			phi_rmu = (phi_rp - phi_rl) / (2.0 * delta_mu)
			
			if (j==1):
				phi_rrr    = (phi[j+2][k] - 2.0 * phi[j+1][k] + phi[j-1][k] + phi[1][k]) / (2.0 * pow(delta_r, 3.0))
			else:
				phi_rrr    = (phi[j+2][k] - 2.0 * phi[j+1][k] + phi[j-1][k] - phi[j-2][k]) / (2.0 * pow(delta_r, 3.0))

#			if (k==m-1):
#				print 'Here m-1: ', k, m-1
#			if (k==m-2):
#				print 'Here m-2: ', k, m-2

			if (k==1):
				phi_mumumu = (phi[j][k+2] - 2.0 * phi[j][k+1] + phi[j][k-1] + phi[j][1]) / (2.0 * pow(delta_mu, 3.0))
			elif (k==m-2):
				phi_mumumu = (-phi[j][k] - 2.0 * phi[j][k+1] + phi[j][k-1] - phi[j][k-2]) / (2.0 * pow(delta_mu, 3.0))
			else:
				phi_mumumu = (phi[j][k+2] - 2.0 * phi[j][k+1] + phi[j][k-1] - phi[j][k-2]) / (2.0 * pow(delta_mu, 3.0))


			phi_rrp1= (phi[j-1][k+1] - 2.0 * phi[j][k+1] + phi[j+1][k+1]) / pow(delta_r, 2.0)
			phi_rrm1= (phi[j-1][k-1] - 2.0 * phi[j][k-1] + phi[j+1][k-1]) / pow(delta_r, 2.0)
			phi_rrmu= (phi_rrp1 - phi_rrm1)/(2.0*delta_mu)

			phi_mumup= (phi[j+1][k-1] - 2.0 * phi[j+1][k] + phi[j+1][k+1]) / pow(delta_mu, 2.0)
			phi_mumul= (phi[j-1][k-1] - 2.0 * phi[j-1][k] + phi[j-1][k+1]) / pow(delta_mu, 2.0)
			phi_rmumu= (phi_mumup - phi_mumul) / (2.0 * delta_r)

			I_r   = (I[j+1][k] - I[j-1][k])/ (2.0 * delta_r)
			I_mu  = (I[j][k+1] - I[j][k-1])/ (2.0 * delta_mu)

			I_rr  = (I[j-1][k] - 2.0 * I[j][k] + I[j+1][k]) / pow(delta_r, 2.0)
			I_mumu= (I[j][k-1] - 2.0 * I[j][k] + I[j][k+1]) / pow(delta_mu, 2.0)

			sigma_r   = (sigma[j+1][k] - sigma[j-1][k])/ (2.0 * delta_r)
			sigma_mu  = (sigma[j][k+1] - sigma[j][k-1])/ (2.0 * delta_mu)



			nabla_I   = [I_r, -I_mu*sintheta/r, 0]
			nabla_fi  = [0.0, 0.0, 1.0/(r*sintheta)]
			nabla_Phi = [phi_r, -phi_mu * sintheta/r, 0] 

#			grad_shafranov   = phi_rr + costheta * phi_mu / r2 + phi_mumu * sintheta2 / r2 - costheta * phi_mu / r2 
#			grad_shafranov_I = I_rr   + costheta * I_mu / r2   + I_mumu   * sintheta * sintheta / r2 - costheta * I_mu /r2

			grad_shafranov   = phi_rr + phi_mumu * sintheta2 / r2  
			grad_shafranov_I = I_rr   + I_mumu   * sintheta2 / r2 


#			print '------------'
#			print grad_shafranov, const1			
#			print '--------------'

#			delta_grad_shafranov1 = phi_rrr - 2.0 * costheta * phi_mu / r3 + costheta * phi_rmu / r2 - 2.0 * sintheta2 * phi_mumu / r3 + sintheta2 * phi_rmumu / r2 + 2.0*costheta * phi_mu /r2 - costheta * phi_rmu /r2
			delta_grad_shafranov1 = phi_rrr - 2.0 * sintheta2 * phi_mumu / r3 + sintheta2 * phi_rmumu / r2 

#			delta_grad_shafranov2 = -sintheta * phi_rrmu / r  - sintheta * costheta * phi_mumu / r3 - sintheta * phi_mu / r3 - sintheta3 *phi_mumumu / r3 + 2.0*sintheta*costheta * phi_mumu /r3 + sintheta * costheta * phi_mumu / r3 + sintheta * phi_mu / r3

			delta_grad_shafranov2 = -sintheta * phi_rrmu / r  - sintheta3 *phi_mumumu / r3 + 2.0*sintheta*costheta * phi_mumu /r3 

			const2 = c / (4.0 * 3.1415926 * e_charge * ne[j][k] * r2 * sintheta2)

			delta_omega1 =  - 2.0 * const2 / r * grad_shafranov             +  const2 * delta_grad_shafranov1
			delta_omega2 =  - 2.0 * const2*costheta /(r * sintheta) * grad_shafranov  +  const2 * delta_grad_shafranov2

			delta_omega_delta_phi_delta_Phi = delta_omega2 * phi_r / ( r * sintheta) + delta_omega1 * phi_mu / r2
	
			const3 = c / (4.0 * 3.1415926 * e_charge * ne[j][k]) 

			delta_hi_delta_phi_delta_I = -const3 * (2.0 * costheta * I_r / (r4 * sintheta4) + 2.0*I_mu / (r4 * r * sintheta2)) 

			delta_I_delta_sigma = I_r * sigma_r + sintheta2 * I_mu * sigma_mu / r2

			delta_phi_delta_t =  - const1 / (ne[j][k]) * (-phi_r * I_mu/r2 + phi_mu*I_r/r2) + c2 / (4.0 * 3.1415926 * sigma[j][k]) * grad_shafranov

#			print '----------------------'
#			print - const1 / ne[j][k] * (phi_r * (-I_mu)/r2 + phi_mu*I_r/r2)
#			print c2 / (4.0 * 3.1415926 * sigma[j][k]) * grad_shafranov
	
			delta_I_delta_t   =  - r2 * sintheta2 * delta_omega_delta_phi_delta_Phi - I[j][k] * delta_hi_delta_phi_delta_I + c*c / (4.0 * 3.1415926 * sigma[j][k]) * ( grad_shafranov_I - delta_I_delta_sigma / sigma[j][k] ) 

#			print '---------------------------------++++'
#			print - r2 * sintheta2 * delta_omega_delta_phi_delta_Phi
#			print '/////////////////////////////////////'
#			print I[j][k] * delta_hi_delta_phi_delta_I
#			print 

#			print '+++++++++++++++++++'
#			print delta_phi_delta_t
#			print delta_I_delta_t

			phi_new[j][k] = phi[j][k] + delta_t * delta_phi_delta_t
			I_new[j][k]   = I[j][k] + delta_t * delta_I_delta_t

	for j in range (0, n):
		phi_new[j][0] = 0.0
		phi_new[j][m-1] = 0.0
		I_new[j][0] = 0.0
		I_new[j][m-1] = 0.0

	for k in range (1, m-1):

#		legendre_fun = lpmn(1, 20, z)[0][1]
		l_polinom = 16
		al = []
		for l in range (1,l_polinom): 
			var = pow(r_star, l) / (l + 1.0) * sqrt(3.1415926*(2.0*l+1.0)) 
			res = 0.0
			for j in range (1, m-1):
				legendre_fun = lpmn(1, l_polinom, 0.5*(mu_grid[j] + mu_grid[j+1]))[0][1]
				res = res + 0.5 * (phi[n-3][j] + phi[n-3][j+1]) * legendre_fun[l] * delta_mu/sqrt(1.0 - pow(mu_grid[j],2.0))
			var = res * var
			al.append(var)
		
		res = 0.0
		legendre_fun = lpmn(1, l_polinom, mu_grid[k])[0][1]
		for l in range (1,l_polinom):
			res = res - al[l-1] / pow(r_star, l+1) * sqrt((2.0*l+1.0) / (4.0*3.1415926)) * sqrt(1.0 - pow(mu_grid[k], 2.0)) * legendre_fun[l]
		res = res * 2 * delta_r


		phi_new[n-1][k] = phi[n-3][k] + res
		phi_new[n-2][k] = phi[n-4][k] + res 

		I_new[n-1][k] = 0.0
		I_new[n-2][k] = 0.0
		I_new[0][k] = 0.0
		phi_new[0][k] = 0.0


	return [phi_new, I_new, delta_t]

n = 40
m = 100


r_grid, mu_grid, phi, I = initial_pure_poloidal (n, m, 10)

plot_res (n, m, phi, I, r_grid, mu_grid)
#plot_res (n, m, I, r_grid, mu_grid)

delta_t = 0

for k in range (0, 7000):
	print k, np.max(phi), np.min(phi), np.max(I), np.min(I), delta_t/3.2e7*k
	phi, I, delta_t = one_time_step (n, m, phi, I, r_grid, mu_grid)
	if (k % 100 == 10):
		plot_res (n, m, phi,I, r_grid, mu_grid)
#		plot_res (n, m, I, r_grid, mu_grid)


print r_grid
print mu_grid
print phi
print I


#print 'Look here -- ', len(r_grid)


#side = np.linspace(-2,2,15)
#X,Y = np.meshgrid(side,side)

#print 'X - ', X

#Z = np.exp(-((X-1)**2+Y**2))

# Plot the density map using nearest-neighbor interpolation

#data = []
#pos  = []
#sigma= []
#for i in range (0, 1000):
#	res, sigma1 = crust_model((0.9 + 0.0001*i)* 1.e6, 0.0)
#	pos1 = 0.9 + 0.0001*i
#	data.append(res)
#	pos.append(pos1)
#	sigma.append(sigma1)
#
#plt.plot(pos, data)
#plt.yscale('log')
#plt.show()
#
#plt.plot(pos, sigma)
#plt.yscale('log')
#plt.show()
#

