import FBODE as FBODE
import fbodesolver as fbodesolver
import tensorflow as tf
import numpy as np


print('\n\ntf.VERSION = ', tf.__version__, '\n\n')
print('\n\ntf.keras.__version__ = ', tf.keras.__version__, '\n\n')

# SEED FOR RANDOMNESS
n_seed = 1



def main():
    # PARAMETERS
    T       =  50.0 #was 40
    E0, I0  =  0.02, 0.03 #was0.03, 0.07
    S0      =  1.0 - I0 - E0
    R0, D0  =  0.0, 0.0
    m0      =  [S0,E0,I0,R0,D0] # initial distribution
    beta    =  0.2 #was0.2
    gamma   =  0.1 # was0.1
    epsilon =  0.15 #was0.2
    rho     =  0.95
    lambda1 =  1.0
    lambda2 =  1.0
    lambda3 =  0.9
    lambda4 =  1.0
    cost_I  =  1.0
    g       =  -0.5
    cost_lambda1 = 2.0
    cost_lambda2 = 2.0
    cost_dead    = 1.0  #Second experiment change 1 (was 1)
    batch_size   = 100
    valid_size   = 200
    n_maxstep    = 10000
    Delta_t      = 0.1  #Second experiment change 3 (was 0.1)

    # SAVE DATA TO FILE
    datafile_name = 'data/data_fbodesolver_params.npz'
    np.savez(datafile_name,
             beta=beta, gamma=gamma, epsilon=epsilon, rho=rho, \
             lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, lambda4=lambda4,
             cost_I=cost_I, cost_lambda1=cost_lambda1, cost_lambda2=cost_lambda2, cost_dead=cost_dead,
             m0=m0, batch_size=batch_size, valid_size=valid_size, n_maxstep=n_maxstep,
             Delta_t=Delta_t, g=g)
    # SOLVE FBODE USING NN
    tf.random.set_seed(n_seed) # ---------- TF2
    tf.keras.backend.set_floatx('float64') # To change all layers to have dtype float64 by default, useful when porting from TF 1.X to TF 2.0  # ---------- TF2
    print("============ BEGIN SOLVER FBODE ============")
    print("================ PARAMETERS ================")
    print("particle #: ",batch_size, "time: ", T, "infected: ", I0, "beta: " , beta, "gamma: ", gamma, "epsilon: ", epsilon, "rho: ", rho,  \
          "c_I: ", cost_I, "c_lambda1: ", cost_lambda1,  "c_lambda2: ", cost_lambda2, "c_dead: ", cost_dead, "max_step: ", n_maxstep, "Delta_t: ", Delta_t)
    print("================ PARAMETERS ================")
    ode_equation = FBODE.FBODEEquation(beta, gamma, rho, epsilon, lambda1, lambda2, lambda3, lambda4, cost_I, cost_lambda1, cost_lambda2, cost_dead, g, Delta_t)
    ode_solver = fbodesolver.SolverODE(ode_equation, T, m0, batch_size, valid_size, n_maxstep) # ---------- TF2
    ode_solver.train()
    # SAVE TO FILE
    print("SAVING FILES...")
    datafile_name = 'data/data_fbodesolver_solution_final.npz'
    np.savez(datafile_name,
                    t_path = ode_solver.t_path,
                    P_path = ode_solver.P_path,
                    U_path = ode_solver.U_path,
                    X_path = ode_solver.X_path,
                    ALPHA_path = ode_solver.ALPHA_path,
                    Z_path = ode_solver.Z_empirical_path,
                    loss_history = ode_solver.loss_history)
    print("============ END SOLVER FBODE ============")


if __name__ == '__main__':
    np.random.seed(n_seed)
    main()
