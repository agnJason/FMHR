import numpy as np
import cv2
import scipy
from scipy.sparse import spdiags
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye as speye


def depth2normal(depth):
    zy, zx = np.gradient(depth)
    ones = np.ones_like(zx)

    normal = np.stack([zx, zy, -ones], axis=2)
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    # norm[norm==0] = 1

    normal = normal / norm

    return normal

def convmtx2(kernel, shape):
    assert kernel.shape[0] == kernel.shape[1]
    k_size = kernel.shape[0]
    h, w = shape
    rows = np.tile(np.arange(k_size), k_size) + (np.arange(k_size).repeat(k_size)*(h+k_size-1))
    step = (np.tile(np.arange(h),w) + np.arange(w*(h+k_size-1), step=h+k_size-1).repeat(h)).repeat(k_size**2)
    rows = np.tile(rows, h*w) + step
    cols = np.arange(h*w).repeat(k_size**2)
    values = np.tile(kernel.flatten('F'), h*w)

    M = csc_matrix((values, (rows,cols)), shape=((h+k_size-1)*(w+k_size-1), h*w))

    return M


def estimate_shading(color_im, normals, color_mask):
    '''
    color_img h w 3
    normals h w 3
    color_mask h w

    returns
    M 4
    s h w
    grayimg h w
    '''

    valid_normals = normals[color_mask]

    c = np.array([0.429043, 0.511664, 0.743125, 0.886227])

    mat = np.stack([c[1] * valid_normals[:,0], c[1] * valid_normals[:,1], c[1] * valid_normals[:,2], c[3] * np.ones_like(valid_normals[:,0])], axis=1) # n * 4


    grayimg = cv2.cvtColor(color_im, cv2.COLOR_BGR2GRAY) / 255

    grayvalue = grayimg[color_mask]

    x = np.linalg.inv(mat.transpose() @ mat) @ mat.transpose() @ grayvalue

    M = np.array([c[1]*x[0], c[1]*x[1], c[1]*x[2], c[3]*x[3]])[:,None] # 4 1

    s = np.zeros_like(color_mask, dtype=np.float64)

    s[color_mask] = (valid_normals@M[:3] + M[3:4]).squeeze()

    return M, s, grayimg



def estimate_albedo_and_specularities(shading, I, depth_map, missing_depth, missing_color, lighting_opt_params):
    h, w = I.shape
    gs_iter = 50


    tau_c, sigma_c, sigma_d, lambda_rho, lambda_beta1, lambda_beta2 =\
    lighting_opt_params['tau_c'], lighting_opt_params['sigma_c'], lighting_opt_params['sigma_d'], lighting_opt_params['lambda_rho'], lighting_opt_params['lambda_beta1'], lighting_opt_params['lambda_beta2']


    Iq = np.pad(I, ((1,1), (1,1)), 'constant', constant_values=0)
    Dq = np.pad(depth_map, ((1,1),(1,1)), 'constant', constant_values=0)

    x, y = np.meshgrid(np.arange(w+2), np.arange(h+2))

    mat_idx = ((x!=0) & (x!=w+1) & (y!=0) & (y!=h+1)).flatten('F')

    length = h * w

    mat = 0
    for r in [-1,0,1]:
        for c in [-1,0,1]:
            if (r==0 and c==0):
                continue

            H = np.zeros([3,3])
            H[r+1, c+1] = -1
            H[1,1] = 1
            H = np.rot90(H, k=2)


            m = convmtx2(H, I.shape) # (h+2)*(w+2) h*w
            m = m[mat_idx,:]

            current_weights = np.zeros_like(I)
            shift_Ip = Iq[r+1:r+h+1, c+1:c+w+1]
            shift_Dp = Dq[r+1:r+h+1, c+1:c+w+1]
            color_diff = (I - shift_Ip)**2
            depth_diff = (depth_map - shift_Dp)**2

            idx = (color_diff < tau_c) & (shift_Ip != 0) & (shift_Dp != 0)

            current_weights[idx] = np.exp(-color_diff[idx] / ( 2 * sigma_c ** 2 ) - depth_diff[idx] / ( 2 * sigma_d ** 2 ) )


            mat = mat + spdiags(current_weights.flatten('F'), 0, length, length) @ m

    valid_samples = (~missing_depth).flatten('F') & (~missing_color).flatten('F')
    mat = mat[:, valid_samples]
    valid_neighbors = ((np.sum(mat, 1) <= 1e-6) & (np.sum(np.abs(mat), 1) != 0)).A.squeeze()
    mat = mat[valid_neighbors,:]
    shading = shading.flatten('F')[valid_samples]
    I = I.flatten('F')[valid_samples]

    length = len(shading)
    S = spdiags(shading**2, 0, length, length)

    big_matrix = S + lambda_rho * mat.T @ mat
    vec = (shading * I)[:,None]

    # lower_triangular = scipy.sparse.tril(big_matrix)
    # upper_triangular = big_matrix - lower_triangular
    # temp_alpha = np.ones_like(vec)
    # lower_triangular = lower_triangular.todense()
    # for i in range(gs_iter):
    #     temp_alpha = inv(lower_triangular) @ (vec - upper_triangular @ temp_alpha)

    temp_alpha = spsolve(big_matrix, vec)


    rho = np.zeros_like(valid_samples, dtype=np.float64)
    rho[valid_samples] = temp_alpha

    Eye = speye(length, length)
    r = rho[valid_samples]

    big_matrix = (1 + lambda_beta2) * Eye + lambda_beta1 * mat.T @ mat
    vec = ( I - r * shading )
    # lower_triangular = scipy.sparse.tril(big_matrix)
    # upper_triangular = big_matrix - lower_triangular
    # temp_beta = np.zeros_like(vec)
    # for i in range(gs_iter):
    #     temp_beta = np.linalg.inv(lower_triangular) @ (vec - upper_triangular @ temp_alpha)

    temp_beta = spsolve(big_matrix, vec)

    beta = np.zeros_like(valid_samples, dtype=np.float64)
    beta[valid_samples] = temp_beta


    return rho.reshape(h,w,order='F'), beta.reshape(h,w,order='F')



def refine_surface(z0, rho, beta, M, I, missing_color, depth_opt_params):
    h, w = I.shape
    lambda_z1, lambda_z2 = depth_opt_params['lambda_z1'], depth_opt_params['lambda_z2']

    x, y = np.meshgrid(np.arange(w+2), np.arange(h+2))
    mat_idx = ((x != 0) & (x != (w+1)) & (y != 0) & (y != (h+1))).flatten('F')

    dx_kernel = np.array([[0,0,0], [-1,1,0], [0,0,0]])
    dx = convmtx2(dx_kernel, (h,w))
    dx = dx[mat_idx]

    dy_kernel = np.array([[0,-1,0], [0,1,0], [0,0,0]])
    dy = convmtx2(dy_kernel, (h,w))
    dy = dy[mat_idx]

    lap_kernel = np.array([[0,1,0], [1,-4,1], [0,1,0]])
    lap = convmtx2(lap_kernel, (h,w))
    valid_lap = ((np.sum(np.abs(lap), 1) != 1) & (np.sum(np.abs(lap), 1) != 0)).A.squeeze()
    lap = lap[valid_lap]

    correction = spdiags(np.sum(lap,1).A.squeeze(), 0, h*w, h*w)
    lap = lap - correction
    
    init_surface = z0.flatten('F')

    interior = ((np.sum(dx,1) == 0) & (np.sum(dy,1) == 0) & (np.sum(np.abs(dx), 1) != 0) & (np.sum(np.abs(dy), 1) != 0)).A.squeeze()
    dx = dx[interior]
    dy = dy[interior]

    valid_dxdy = (np.abs(dx@init_surface) < 10) & (np.abs(dy@init_surface) < 10)
    dx = dx[valid_dxdy]
    dy = dy[valid_dxdy]

    color_idx = (~missing_color).flatten('F')
    color_idx = color_idx[interior]
    color_idx = color_idx[valid_dxdy]

    I_s = I.flatten('F')[interior]
    I_s = I_s[valid_dxdy]
    I_s = I_s[color_idx]

    rho_s = rho.flatten('F')[interior]
    rho_s = rho_s[valid_dxdy]
    rho_s = rho_s[color_idx]    

    betas_s = beta.flatten('F')[interior]
    betas_s = betas_s[valid_dxdy]
    beta_s = betas_s[color_idx]

    dx = dx[color_idx]
    dy = dy[color_idx]

    lap_idx = ((np.sum(lap, 1) == 0) & (np.sum(np.abs(lap), 1) != 0)).A.squeeze()
    lap = lap[lap_idx]
    valid_lap = np.abs(lap@init_surface) < 50
    lap = lap[valid_lap]

    z = init_surface
    length = len(rho_s)
    Alpha = spdiags(rho_s, 0, length, length)

    p = dx @ init_surface
    q = dy @ init_surface
    eta = 1. / np.sqrt(1 + p**2 + q**2)
    
    # nx = -eta * p
    # ny = -eta * q
    # nz = -eta
    # E1 = (rho_s * (M[0] * nx + M[1] * ny + M[2] * nz + M[3]) + beta_s - I_s)

    # E2 = np.sqrt(lambda_z1) * (z - init_surface)
    # E3 = np.sqrt(lambda_z2) * lap @ z

    # energy = np.stack([E1,E2,E3], axis=1)


    Eta = spdiags(eta, 0, length, length)
    W = -Alpha @ Eta @ (M[0] * dx + M[1] * dy)
    const = - (beta_s + rho_s * (M[3] - eta * M[2]) - I_s)
    big_matrix = (W.T @ W) + lambda_z1 * speye(h*w) + lambda_z2 * (lap.T @ lap)
    vec = W.T @ const + lambda_z1 * init_surface
    # lower_triangular = scipy.sparse.tril(big_matrix)
    # upper_triangular = big_matrix - lower_triangular
    # gs_iter = 20
    # temp_z = z.copy()

    # for i in range(gs_iter):
    #     temp_z = np.linalg.inv(lower_triangular) @ (vec - upper_triangular @ temp_z)

    temp_z = spsolve(big_matrix, vec)

    # p = dx @ temp_z
    # q = dy @ temp_z
    # eta = 1 / np.sqrt(1 + p**2 + q**2)
    # nx = -eta * p
    # ny = -eta * q
    # nz = -eta
    # E1 = (rho_s * (M[0] * nx + M[1] * ny + M[2] * nz + M[3]) + betas_s - I_s)
    # E2 = np.sqrt(lambda_z1) * (temp_z - init_surface)
    # E3 = np.sqrt(lambda_z2) * lap @ temp_z
    # prev_energy = energy

    # energy = np.stack([E1,E2,E3])


    return temp_z.reshape(h,w,order='F')


def refine_normals(normal_map, rho, beta, M, I, mask, depth_opt_params):
    valid_normals = normal_map[mask]
    valid_rho = rho[mask]
    valid_beta = beta[mask]
    valid_I = I[mask]

    