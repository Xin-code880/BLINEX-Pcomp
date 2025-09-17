import numpy as np
import numpy.linalg as nl
from sklearn.metrics.pairwise import pairwise_kernels
from copy import deepcopy

def training(args, X, X1, C, a, b, C2, m, lamda, mu, pai_plus, pai_minus, eps):
    max_iter = 10
    max_iter_gamma= 20
    max_iter_alpha= 1000

    n = X[0].shape[0]

    K_hat_list = []
    if args.kernel == 'linear':
        for x, x1 in zip(X, X1):
            K = pairwise_kernels(x, x, metric='linear')
            K1 = pairwise_kernels(x, x1, metric='linear')
            K2 = pairwise_kernels(x1, x1, metric='linear')
            k_hat = np.block([[K, K1], [K1.T, K2]])
            K_hat_list.append(k_hat)
    elif args.kernel == 'rbf':
        for x,x1 in zip(X, X1):
            K = pairwise_kernels(x, x, metric='rbf', gamma = args.g)
            K1 = pairwise_kernels(x, x1, metric='rbf', gamma = args.g)
            K2 = pairwise_kernels(x1, x1, metric='rbf', gamma = args.g)
            k_hat = np.block([[K, K1], [K1.T, K2]])
            K_hat_list.append(k_hat)


    norm_K = np.array([nl.norm(K_hat_list[v]) for v in range(m)])# tanglong0714

    
    D = a * b * C * n
    P = 2 * C2 * n

    # initialize
    alpha_star_list = [np.zeros((2 * n, 1)) for v in range(m)]
    grad_s_list = [np.zeros((2 * n, 1)) for v in range(m)]
    alpha_hat_list = [np.zeros((2 * n, 1)) for v in range(m)]
    gamma = np.full((m, 1), 1/m)
    tau = gamma
    
    grad_list = [np.zeros((2 * n, 1)) for v in range(m)]# tanglong


    F = np.zeros((m + 1, m))
    F[0, :] = 1
    np.fill_diagonal(F[1:], 1)

    H=np.block([[np.zeros((1,m))],[-np.eye(m)]])
   
    d = np.zeros((m + 1, 1))
    d[0] = 1

    M = np.zeros((m + 1, 1))

    count = 0
    while True:
        count_gamma = 0
        count_alpha = 0
        count += 1
        print('count:',count)
        zeta = np.array([alpha_hat_list[v].T @ K_hat_list[v] @ alpha_hat_list[v] for v in range(m)]).reshape(-1, 1)
        ogamma=gamma.copy()
    
        while True:
            count_gamma += 1
            # update gamma
            ogamma_t = gamma.copy()
            gamma = -nl.inv(lamda * np.eye(m) + mu * F.T @ F) @ (zeta / 2 + mu * F.T @ (H @ tau - d + M / mu))
            gamma_norm = nl.norm(gamma - ogamma_t)

            # update tau
            otau = tau.copy()
            tau = -nl.inv(H.T @ H) @ H.T @ (F @ gamma - d + M / mu)
            tau = (tau > 0) * tau
            tau_norm = nl.norm(tau - otau)

            M = M + mu * (F @ gamma + H @ tau - d)

            mu = mu * 1.05
            
            c_norm=nl.norm(F @ gamma + H @ tau - d)
            
            value_max = max(gamma_norm, tau_norm,c_norm)
            #print('count gamma','%d:%f,%f,%f' % (count_gamma,gamma_norm,tau_norm,c_norm))
            if value_max <= eps or count_gamma >= max_iter_gamma:
                print('count gamma', '%d:%f,%f,%f' % (count_gamma, gamma_norm, tau_norm, c_norm))
                break

        
        # update alpha
        oalpha_hat_list=deepcopy(alpha_hat_list)
        while True:
            count_alpha += 1
            ograd_list=deepcopy(grad_list)
            for v in range(m):
                Hi = np.zeros((2 * n, 2 * n))
                K_hat = K_hat_list[v]
                alpha_hat = alpha_hat_list[v]
                alpha_star = alpha_star_list[v]
                gamma_v = gamma[v]

                # g1, h1, g2, h2
                g1 = (alpha_hat.T @ K_hat.T[:, 0:n] < 1 - 1e-6) * (-K_hat.T[:, 0:n])
                h1 = (alpha_hat.T @ K_hat.T[:, n:2 * n] < 1 - 1e-6) * (pai_minus * K_hat.T[:, n:2 * n])
                g2 = (alpha_hat.T @ K_hat.T[:, n:2 * n] > -1 + 1e-6) * (K_hat.T[:, n:2 * n])
                h2 = (alpha_hat.T @ K_hat.T[:, 0:n] > -1 + 1e-6) * (-pai_plus * K_hat.T[:, 0:n])

                # calcuate kesi and eta
                kesi = (alpha_hat.T @ K_hat.T[:, 0:n] < 1 - 1e-6) * (1 - alpha_hat.T @ K_hat.T[:, 0:n]) \
                       - (alpha_hat.T @ K_hat.T[:, n:2 * n] < 1 - 1e-6) * (
                               1 - alpha_hat.T @ K_hat.T[:, n:2 * n]) * pai_minus
                yita = (alpha_hat.T @ K_hat.T[:, n:2 * n] > -1 + 1e-6) * (1 + alpha_hat.T @ K_hat.T[:, n:2 * n]) \
                       - (alpha_hat.T @ K_hat.T[:, 0:n] > -1 + 1e-6) * (1 + alpha_hat.T @ K_hat.T[:, 0:n]) * pai_plus

                G = (g1 + h1) * ((np.clip(np.exp(a * kesi), a_min=-20, a_max=20) - 1) / (
                            1 + b * (np.clip(np.exp(a * kesi), a_min=-20, a_max=20) - a * kesi - 1)) ** 2) \
                    + (g2 + h2) * ((np.clip(np.exp(a * yita), a_min=-20, a_max=20) - 1) / (
                            1 + b * (np.clip(np.exp(a * yita), a_min=-20, a_max=20) - a * yita - 1)) ** 2)

                # aggerate all views gradients
                for u in range(m):
                    if v != u:
                        K_v = K_hat_list[v]
                        K_u = K_hat_list[u]
                        alpha_v = alpha_hat_list[v]
                        alpha_u = alpha_hat_list[u]

                        Hi += (alpha_v.T @ K_v) * K_v - (alpha_u.T @ K_u) * K_v
                
                
                Hi=Hi[:,:n]+Hi[:,n:]#tanglong 0713
                

                # calcuate gradient
                grad_list[v] = gamma_v * K_hat @ alpha_hat + C * a * b * G.sum(1).reshape(-1, 1) + 2 * C2 * Hi.sum(1).reshape(-1, 1)
                grad_s_list[v] = grad_s_list[v] + grad_list[v] * count_alpha / 2

                # calcuate gradient norm
                G_grad_norm = np.abs(np.diag((g1 + h1).T @ (g1 + h1)) * (
                        a * np.exp(a * kesi) * (
                            1 + b * (np.exp(a * kesi) - a * kesi - 1)) - 2 * a * b * (
                                np.exp(a * kesi) - 1) ** 2) \
                                     / (1 + b * (np.exp(a * kesi) - a * kesi - 1)) ** 3 \
                                     + np.diag((g2 + h2).T @ (g2 + h2)) * (
                                                 a * np.exp(a * yita) * (
                                                 1 + b * (np.exp(a * yita) - a * yita - 1)) - 2 * a * b * (
                                                             np.exp(a * yita) - 1) ** 2) \
                                     / (1 + b * (np.exp(a * yita) - a * yita - 1)) ** 3)
                G_grad_norm = G_grad_norm.reshape(1, -1)

                if v==0:
                    G_grad_norm_T=G_grad_norm
                else:
                    G_grad_norm_T=np.append(G_grad_norm_T, G_grad_norm, axis=0)
                    

                H_grad_norm = (m - 1) * gamma_v * np.abs(np.diag(K_hat_list[v][:,:n].T @ K_hat_list[v][:,:n]))\
                    +(m - 1) * gamma_v * np.abs(np.diag(K_hat_list[v][:,n:].T @ K_hat_list[v][:,n:]))
                H_grad_norm = H_grad_norm.reshape(1,-1)
                
                for u in range(m):
                    if u != v:
                        temp=np.abs(np.diag(-K_hat_list[u][:,:n].T @ K_hat_list[v][:,:n]))\
                                                +np.abs(np.diag(-K_hat_list[u][:,n:].T @ K_hat_list[v][:,n:]))
                        H_grad_norm = np.append(H_grad_norm,temp.reshape(1,-1), axis=0)
                
                if v==0:
                    H_grad_norm_T=H_grad_norm
                else:
                    H_grad_norm_T=np.append(H_grad_norm_T, H_grad_norm, axis=0)
                
            L_k = np.sqrt((norm_K**2).sum(0)) + D * np.sqrt((G_grad_norm_T**2).sum(0)).max(0) \
                + P * np.sqrt((H_grad_norm_T**2).sum(0)).max(0)
                
            for v in range(m):
                v_k = alpha_hat - 1 / L_k * grad_list[v]
                z_k = alpha_star - 1 / L_k * grad_s_list[v]
                alpha_hat_list[v] = 2 / (count_alpha + 3) * z_k + (count_alpha + 1) / (count_alpha + 3) * v_k
                #update(count_alpha,v,nl.norm(grad_list[v]))

            # alpha comvergence
            grad_diff_norm = [nl.norm(grad_list[v]-ograd_list[v]) for v in range(m)]
            #K_hat_norm = [nl.norm(K_hat_list[v]) for v in range(m)]
            grad_norm = [nl.norm(grad_list[v]) for v in range(m)]
            #print('count_alpha:', count_alpha, max(grad_diff_norm)/max(grad_norm))
            grad_norm_final = max(grad_diff_norm) / max(grad_norm)
            if grad_norm_final<=eps or np.isnan(grad_norm_final) or count_alpha>max_iter_alpha:
                print('count_alpha:', count_alpha, grad_norm_final)
                break

        #Overall convergence
        gamma_diff = nl.norm(gamma-ogamma)
        alpha_diff = [nl.norm(alpha_hat_list[v]- oalpha_hat_list[v]) for v in range(m)]
        print('gamma_diff:', gamma_diff)
        print('alpha_diff:', alpha_diff)
        print('gamma_diff:',gamma_diff,'alpha_diff:',alpha_diff)
        if (gamma_diff <= eps and max(alpha_diff)<eps) or np.isnan(grad_norm_final) or count >= max_iter:
            if np.isnan(grad_norm_final):
                print("nan")
            print('gamma_diff:', gamma_diff, 'alpha_diff:', alpha_diff)
            print('sucessfully')
            break

    alpha = [alpha_hat_list[v][0:n, ] for v in range(m)]
    alpha1 = [alpha_hat_list[v][n:2 * n, ] for v in range(m)]
    return alpha, alpha1 ,gamma



