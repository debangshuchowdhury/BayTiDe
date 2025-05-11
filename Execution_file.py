import numpy as np
from multiple_delay import Gibbs
from single_delay import Gibbs as single_Gibbs


np.random.seed(10000)

X = np.genfromtxt('xdat.csv', delimiter=',').reshape(-1,1)
print('X shape = ', X.shape)
XDOT = np.genfromtxt('xdotdat.csv', delimiter=',') .reshape(-1,1)
t_final = 100

t = np.linspace(0, t_final, X.shape[0])

NUM_DELAYS = 1
NUM_VARIABLES = X.shape[-1]
NUM_TARGETS = 1
NUM_TIMESTEPS = X.shape[0]
DT = t[1] - t[0]
print('dt = ', DT)

functions = [lambda x: x,
             lambda x: x**2,
             lambda x,y: x*y,
             lambda x: x**3,
             lambda x,y: x*y**2,
             lambda x,y: (x**2)*y,
             lambda x: x**4,
             lambda x,y: (x**3)*y,
             lambda x,y: (y**3)*x,
             lambda x,y: (x**2)*(y**2),
             lambda x: x**5,
             lambda x,y: (x**4)*(y),
             lambda x,y: (x)*(y**4),
             lambda x,y: (x**3)*(y**2),
             lambda x,y: (x**2)*(y**3),
             lambda x: x**6,
             lambda x,y: (x**5)*(y),
             lambda x,y: (x)*(y**5),
             lambda x,y: (x**4)*(y**2),
             lambda x,y: (x**2)*(y**4),
             lambda x,y: (x**3)*(y**3),#, # 27
            #  lambda x: np.exp(-x),
            #  lambda x: np.exp(x),
             lambda x: np.sin(x),
             lambda x: np.cos(x)]
            #  lambda x: 1/x,
            #  lambda x: 1/x**2]
            #  lambda x,y: 1/(x*y),
            #  lambda x: 1/(x**3),
            #  lambda x,y: 1/(x*y**2),
            #  lambda x,y: 1/(y*x**2)]
# functions = [ lambda x: x,
#              lambda x: x**2,
#              lambda x: np.sin(x),
#              lambda x: np.cos(x),
#              lambda x: x/(1+x**10),
#              lambda x: x/(1+x**4),
#              lambda x: 1/x,
#              lambda x: 1/x**2]

# functions = [lambda x: x]
#             #  lambda x: x**2]

NUM_SAMPLES = 500
BURN_IN = 100 #int(NUM_SAMPLES/3)

START_INDICES = np.array([50])
END_INDICES = np.array([1000])
THRESHOLD = 1e-10


# MODEL = Gibbs(NUM_DELAYS, NUM_VARIABLES, NUM_TARGETS, NUM_TIMESTEPS, DT, functions, start_indices=START_INDICES, end_indices=END_INDICES, num_samples=NUM_SAMPLES, burn_in=BURN_IN, G_threshold=THRESHOLD)
MODEL = single_Gibbs(NUM_DELAYS, NUM_VARIABLES, NUM_TARGETS, NUM_TIMESTEPS, DT, functions=functions, start_indices=START_INDICES, end_indices=END_INDICES, num_samples=NUM_SAMPLES, burn_in=BURN_IN, G_threshold=THRESHOLD)

final_weights, final_latent, sigma_weights, final_tau, function_names, theta, taus = MODEL.forward(X, XDOT, verbose=True, verbose_interval=1)

print('final weights = ', final_weights)
print('final latent = ', final_latent)
print('final tau = ', final_tau)
print("function names = ", function_names, "\n\n")
lat = final_latent>0.5
print('relevant weights = ', final_weights[lat])
print('final functions = ', function_names[lat])
print('sigma weights = ', sigma_weights[lat])
# np.savetxt('coupled_thetas_2.csv', theta[lat], delimiter=',')
# np.savetxt('sampled_taus_coupled0.csv', taus, delimiter=',')
np.savetxt('nodelayf_weights.csv', final_weights[lat], delimiter=',')
np.savetxt('nodelay_library_taus.csv', taus, delimiter=',')
np.savetxt('nodelay_library_pip.csv', final_latent, delimiter=',')
# np.savetxt('jcsprott_increased_library_funcs_exp.csv', function_names[lat], delimiter=',')