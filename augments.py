import torch
import torch.nn as nn

# Based on https://gist.github.com/catalys1/3eef0b6ee5749b5d8851755a7ee0e6e1

def get_tensortype(tensor):
    '''Return dict containing dtype and device of `tensor`, which can be passed as
    **kwargs to tensor construction functions.
    '''
    return dict(dtype=tensor.dtype, device=tensor.device)

def pair_square_euclidean(x1, x2):
    '''Compute the pairwise squared euclidean distance matrix between points defined in `x1` and `x2`.
    '''
    # ||x1 - x2||^2 = (x1-x2)^T(x1-x2) = x1^T*x1 + x2^T*x2 - 2*x1^T*x2
    x1_sq = x1.mul(x1).sum(dim=-1, keepdim=True)
    x2_sq = x2.mul(x2).sum(dim=-1, keepdim=True).transpose(1,2)
    x1_x2 = x1.matmul(x2.transpose(1,2))
    square_dist = -2 * x1_x2 + x1_sq + x2_sq
    square_dist = square_dist.clamp(min=0)  # handle possible numerical errors
    return square_dist

def kernel_distance(r_sq, eps=1e-8):
    '''Compute the TPS kernel distance function: r^2*log(r), where r is the euclidean distance.
    Since log(r) = 1/2*log(r^2), this function takes the squared distance matrix `r_sq` and calculates
    0.5 * r_sq * log(r_sq).
    '''
    # r^2 * log(r) = 1/2 * r^2 * log(r^2)
    return 0.5 * r_sq * r_sq.add(eps).log()

def get_tps_parameters(source_points, dest_points):
    '''Computes the TPS transform parameters that warp `source_points` to `dest_points`.
    Returns a tuple (rbf_weights, affine_weights).
    '''
    # source_points and dest_points have the same shape: (batch_size, num_points, 2), with the
    # last dimension being (x,y) coordinates
    tensortype = get_tensortype(source_points)
    batch_size, num_points = source_points.shape[:2]
    
    # set up and solve linear system
    # [K P; P^T 0] [w; a] = [dst; 0]
    pair_distance = pair_square_euclidean(source_points, dest_points)
    k_matrix = kernel_distance(pair_distance)
    
    dest_with_zeros = torch.cat((dest_points, torch.zeros(batch_size, 3, 2, **tensortype)), 1)
    p_matrix = torch.cat((torch.ones(batch_size, num_points, 1, **tensortype), source_points), -1)
    p_matrix_t = torch.cat((p_matrix, torch.zeros(batch_size, 3, 3, **tensortype)), 1).transpose(1,2)
    l_matrix = torch.cat((k_matrix, p_matrix), -1)
    l_matrix = torch.cat((l_matrix, p_matrix_t), 1)
    weights = torch.linalg.solve(l_matrix, dest_with_zeros)
    rbf_weights = weights[:, :-3]
    affine_weights = weights[:, -3:]
    
    return rbf_weights, affine_weights

def tps_warp_points(source_points, kernel_points, rbf_weights, affine_weights):
    '''Given a set of source points, kernel points (the destination points from solving for the TPS weights),
    and the rbf and affine weights of the TPS transform: warps the source points using the TPS transform.
    '''
    # f_{x|y}(v) = a_0 + [a_x a_y].v + \sum_i w_i * U(||v-u_i||)
    pair_distance = pair_square_euclidean(source_points, kernel_points)
    k_matrix = kernel_distance(pair_distance)
    
    # broadcast the kernel distance matrix against the x and y weights to compute the x and y
    # transforms simultaneously
    warped = (k_matrix[...,None].mul(rbf_weights[:,None]).sum(-2) + 
              source_points[...,None].mul(affine_weights[:,1:]).sum(-2) + 
              affine_weights[:,0])
    
    return warped

class TPSWarp(nn.Module):
    def __init__(self):
        super().__init__()
        xs = torch.tensor([-1.0, 1, 0, -1, 1])
        ys = torch.tensor([-1.0, -1, 0, 1, 1])
        self.register_buffer("points", torch.stack((xs, ys), dim=1)[None])
    
    def forward(self, img:torch.Tensor):
        dst = self.points + torch.rand_like(self.points)*.6 -.25
        rbf_weights, affine_weights = get_tps_parameters(dst, self.points)
        batch_size, _, h, w = img.shape
        dtype, device = self.points.dtype, self.points.device
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, dtype=dtype, device=device),
                                torch.linspace(-1, 1, w, dtype=dtype, device=device))
        coords = torch.stack([xs, ys], -1).view(-1, 2)
        coords = torch.stack([coords]*batch_size, 0)
        warped = tps_warp_points(coords, self.points, rbf_weights, affine_weights).view(-1, h, w, 2)
        warped_image = torch.nn.functional.grid_sample(img, warped, align_corners=False)
        return warped_image