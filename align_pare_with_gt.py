import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import cv2
from dataset.behave_dataset_pare import BehaveImgDataset
from torch.optim import Adam
import wandb
import json

wandb.init(project = "PARE Alignment")


def show(img):
    figure(figsize=(8, 6), dpi=80)
    plt.imshow(img)
    plt.show()

def show_projection(ver, img):
    for i in range(ver.shape[0]):
        img = cv2.circle(img, (ver[i, 0].int().item(), ver[i, 1].int().item()), 2, (255, 0, 0), 1)
    show(img)

def calc_error(S1, S2):
    return torch.sqrt(((S1 - S2) ** 2).sum(dim=-1)).mean()


def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t # why this scale is applied directory to points?

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def compute_transform(S1, S2):
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    return R, t, scale, transposed

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

class Model(nn.Module):
    def __init__(self, source_verts, source_camera, inti, dest_img=None, Trans=True, Rot=True):
        super().__init__()
        
        dest_cal_matrix, dest_dist_coefs = inti

        self.source_verts = source_verts.to('cuda')
        self.source_camera = source_camera
        self.dest_cal_matrix = dest_cal_matrix
        self.dest_dist_coefs = dest_dist_coefs
        self.b_img = dest_img.permute(1, 2, 0).numpy()

        self.pare_projection = self.get_src_projection()
        
        self.Trans = Trans
        if self.Trans:
            self.trans = torch.nn.parameter.Parameter(data=torch.tensor([[0.0, 0.0, 2.0]]), requires_grad=True)
            
        self.Rot = Rot
        if self.Rot:
            self.d6 = torch.nn.parameter.Parameter(data=torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]), requires_grad=True)

    def get_src_projection(self):
        sx, sy, tx, ty = self.source_camera
        verts = self.source_verts.cpu().clone()
        verts[:, 1:] *= -1

        verts += torch.tensor([[0, 0, -1000]])

        P = torch.eye(4).float()
        P[0, 0] = sx
        P[1, 1] = sy
        P[0, 3] = tx * sx
        P[1, 3] = -ty * sy
        P[2, 2] = -1

        verts = torch.cat([verts, torch.ones((verts.shape[0], 1))], dim=-1)
        verts = (P.float() @ verts.float().T).T

        verts = verts[:, :3] / verts[:, 2:3]
        verts = verts[:, :2] * 1000
        verts[:, 1] *= -1

        verts = verts + torch.tensor([[0.5, 0.5]]) * 2
        verts = verts * torch.tensor([self.b_img.shape[1], self.b_img.shape[0]]) / 2
        
        return verts.to('cuda')
    
    def get_dest_projection(self, ver):
        calibration_matrix = torch.tensor(self.dest_cal_matrix).float().to('cuda')
        k1, k2, p1, p2, k3, k4, k5, k6 = self.dest_dist_coefs

        res = torch.zeros_like(ver)

        ver = ver[:, :2] / ver[:, 2:3]
        x_p = ver[:, 0]
        y_p = ver[:, 1]

        r2 = x_p ** 2 + y_p ** 2
        c = (1 + k1*r2 + k2*(r2**2) + k3*(r2**3)) / (1 + k4*r2 + k5*(r2**2) + k6*(r2**3))

        res[:, :2] = ver * c.unsqueeze(1)
        res[:, 0] += 2 * p1 * x_p * y_p + p2
        res[:, 1] += 2 * p2 * x_p * y_p + p1 * (r2 + 2*(y_p ** 2))

        res = calibration_matrix @ res.T
        res = res.T
        res = res[:, :2]

        res += torch.tensor([self.b_img.shape[1], self.b_img.shape[0]]).to('cuda') / 2
        return res

    def get_moved_pare(self):
        res = self.source_verts.clone()
        if self.Rot:
            res = res @ rotation_6d_to_matrix(self.d6)[0]
        if self.Trans:
            res = res + self.trans
        return res

    def forward(self, show=False):
        
        res = self.get_dest_projection(self.get_moved_pare())

        er = torch.mean(torch.norm(res - self.pare_projection, dim=-1) ** 2)

        if show:
            show_projection(res, self.b_img.copy())
            show_projection(self.pare_projection, self.b_img.copy())

        return er
    
if __name__ == "__main__":

    # make an options object
    class Options():
        def __init__(self):
            self.max_dataset_size = 5000000
            self.trunc_thres = 0.2

    opt = Options()

    dataset = BehaveImgDataset()
    dataset.initialize(opt, 'test', load_extra=True)

    aligned_pare_dict = {}

    for i in range(2):
        out = dataset[i]

        print(out['img_path'])

        intrin = (
            out['calibration_matrix'],
            out['dist_coefs']
        )

        model2 = Model(out['pare_verts'].clone(), out['pare_camera'], intrin, out['img'], True, True).to('cuda')

        optimizer = Adam(model2.parameters(), lr=0.1)

        for i in range(500):
            optimizer.zero_grad()
            loss = model2()
            loss.backward()
            optimizer.step()
            # if i % 500 == 0:
            #     print(i, loss.detach().item())
        print(loss.detach().item())

        behave_verts = out['body_mesh_verts'].clone().cpu().float()
        pare_verts = out['pare_verts'].clone().cpu()

        aligned_pare = compute_similarity_transform(pare_verts.T.numpy(), behave_verts.T.numpy())
        aligned_pare = torch.from_numpy(aligned_pare.T).float()
        # moved_pare1 = model1.get_moved_pare().detach().cpu()
        moved_pare2 = model2.get_moved_pare().detach().cpu()

        aligned_er = calc_error(behave_verts, aligned_pare)
        # moved_er1 = calc_error(behave_verts, moved_pare1)
        moved_er2 = calc_error(behave_verts, moved_pare2)

        print(aligned_er)
        # print(moved_er1)
        print(moved_er2)

        day_key = out['img_path'][len('/data/xiwang/behave/sequences/'):]
        aligned_pare_dict[day_key] = moved_pare2.numpy().tolist()

    with open('/data/aruzzi/Behave/aligned_pare', 'w') as f:
        json.dump(aligned_pare_dict, f)

