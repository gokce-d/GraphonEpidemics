import FBODE as FBODE
import fbodesolver as fbodesolver
import tensorflow as tf
import numpy as np


print('\n\ntf.VERSION = ', tf.__version__, '\n\n')
print('\n\ntf.keras.__version__ = ', tf.keras.__version__, '\n\n')

# SEED FOR RANDOMNESS
n_seed = 7


def main():
    # PARAMETERS
    batch_size = 100
    T =          40.0
    I0, R0 =     0.05, 0.0
    S0 =         1.0 - I0 - R0
    m0 =         [S0,I0,R0] # initial distribution
    beta =       [0.4, 0.4, 0.3]
    gamma =      0.1
    kappa =      0.0
    lambda1 =    1.0
    lambda2 =    0.9
    lambda3 =    1.0
    cost_I =     1.0
    cost_lambda1 = 10.0
    valid_size =   200
    n_maxstep =    20000
    g =            0.2
    Delta_t =      0.05

    # SAVE DATA TO FILE
    datafile_name = 'data/data_fbodesolver_params.npz'
    np.savez(datafile_name,
                beta=beta, gamma=gamma, kappa=kappa, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
                cost_I=cost_I, cost_lambda1=cost_lambda1,
                m0=m0,
                batch_size=batch_size, valid_size=valid_size, n_maxstep=n_maxstep)
    # SOLVE FBODE USING NN
    tf.random.set_seed(n_seed) # ---------- TF2
    tf.keras.backend.set_floatx('float64') # To change all layers to have dtype float64 by default, useful when porting from TF 1.X to TF 2.0  # ---------- TF2
    print("============ BEGIN SOLVER FBODE ============")
    print("================ PARAMETERS ================")
    print("particle #: ",batch_size, "time: ", T, "infected: ", I0, "beta: " , beta, "gamma: ", gamma, "kappa: ", kappa, "c_I: ", cost_I, "c_lambda: ", cost_lambda1, "max_step: ", n_maxstep, "Delta_t: ", Delta_t)
    print("================ PARAMETERS ================")
    ode_equation = FBODE.FBODEEquation(beta, gamma, kappa, lambda1, lambda2, lambda3, cost_I, cost_lambda1, g, Delta_t)
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
