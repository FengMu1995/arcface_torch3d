import math
import torch
from torch import nn
from . import Auxbackbone
from .renderer import Renderer



class AuxiRecon(nn.Module):
    def __init__(self, cfg_auxi=None):
        super(AuxiRecon, self).__init__()
        self.image_size = cfg_auxi.get('image_size', 64)
        self.downsample = nn.Upsample(size=(self.image_size, self.image_size), mode='bilinear', align_corners=True)
        self.min_depth = cfg_auxi.get('min_depth', 0.9)
        self.max_depth = cfg_auxi.get('max_depth', 1.1)
        self.border_depth = cfg_auxi.get('border_depth', (0.7 * self.max_depth + 0.3 * self.min_depth))
        self.min_amb_light = cfg_auxi.get('min_amb_light', 0.)
        self.max_amb_light = cfg_auxi.get('max_amb_light', 1.)
        self.min_diff_light = cfg_auxi.get('min_diff_light', 0.)
        self.max_diff_light = cfg_auxi.get('max_diff_light', 1.)
        self.xyz_rotation_range = cfg_auxi.get('xyz_rotation_range', 60)
        self.xy_translation_range = cfg_auxi.get('xy_translation_range', 0.1)
        self.z_translation_range = cfg_auxi.get('z_translation_range', 0.1)
        self.renderer = Renderer(cfg_auxi)

        ## Auxbackbone and optimizers
        # IIA
        self.netL = Auxbackbone.Encoder(cin=3, cout=4, nf=32)
        self.netV = Auxbackbone.Encoder(cin=3, cout=6, nf=32)
        # IRA
        self.netD = Auxbackbone.EDDeconv(cin=256, cout=1, nf=64, zdim=256, activation=None)
        self.netA = Auxbackbone.EDDeconv(cin=256, cout=3, nf=64, zdim=256)

        self.depth_rescaler = lambda d: (1 + d) / 2 * self.max_depth + (1 - d) / 2 * self.min_depth
        self.amb_light_rescaler = lambda x: (1 + x) / 2 * self.max_amb_light + (1 - x) / 2 * self.min_amb_light
        self.diff_light_rescaler = lambda x: (1 + x) / 2 * self.max_diff_light + (1 - x) / 2 * self.min_diff_light

    def forward(self, f1, f2):
        b, _, _, _ = f1.shape
        recon_in = self.downsample(f1.float())
        ## predict lighting
        canon_light = self.netL(recon_in)  # Bx4
        self.canon_light_a = self.amb_light_rescaler(canon_light[:, :1])  # ambience term
        self.canon_light_b = self.diff_light_rescaler(canon_light[:, 1:2])  # diffuse term
        canon_light_dxy = canon_light[:, 2:]  # Bx2
        self.canon_light_d = torch.cat([canon_light_dxy, torch.ones(b, 1).to(canon_light_dxy.device)], 1)  # Bx3
        self.canon_light_d = self.canon_light_d / (
            (self.canon_light_d ** 2).sum(1, keepdim=True)) ** 0.5  # diffuse light direction

        ## predict viewpoint transformation
        self.view = self.netV(recon_in)
        self.view = torch.cat([
            self.view[:, :3] * math.pi / 180 * self.xyz_rotation_range,
            self.view[:, 3:5] * self.xy_translation_range,
            self.view[:, 5:] * self.z_translation_range], 1)

        ## predict canonical depth
        self.canon_depth_raw = self.netD(f2.float()).squeeze(1)  # BxHxW
        self.canon_depth = self.canon_depth_raw - self.canon_depth_raw.view(b, -1).mean(1).view(b, 1, 1)
        self.canon_depth = self.canon_depth.tanh()
        self.canon_depth = self.depth_rescaler(self.canon_depth)
        ## clamp border depth
        depth_border = torch.zeros(1, self.image_size, self.image_size-4).to(self.canon_depth.device)
        depth_border = nn.functional.pad(depth_border, (2, 2), mode='constant', value=1)
        self.canon_depth = self.canon_depth * (1 - depth_border) + depth_border * self.border_depth

        ## predict canonical albedo
        self.canon_albedo = self.netA(f2.float())  # Bx3xHxW
        ## shading
        self.canon_normal = self.renderer.get_normal_from_depth(self.canon_depth)
        self.canon_diffuse_shading = (self.canon_normal * self.canon_light_d.view(-1, 1, 1, 3)).sum(3).clamp(
            min=0).unsqueeze(1)
        canon_shading = self.canon_light_a.view(-1, 1, 1, 1) + self.canon_light_b.view(-1, 1, 1,
                                                                                       1) * self.canon_diffuse_shading
        self.canon_im = (self.canon_albedo / 2 + 0.5) * canon_shading * 2 - 1


        ## reconstruct input view
        self.renderer.set_transform_matrices(self.view)
        self.recon_depth = self.renderer.warp_canon_depth(self.canon_depth)
        grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(self.recon_depth)
        self.recon_im = nn.functional.grid_sample(self.canon_im, grid_2d_from_canon, mode='bilinear',
                                                  align_corners=True)
        margin = (self.max_depth - self.min_depth) / 2
        recon_im_mask = (self.recon_depth < self.max_depth + margin).float().unsqueeze(1).detach()  # invalid border pixels have been clamped at max_depth+margin
        self.recon_im = self.recon_im * recon_im_mask
        vis_depth = (self.recon_depth - self.min_depth) / (self.max_depth - self.min_depth)
        vis_albedo = self.canon_albedo
        return self.recon_im, recon_in, recon_im_mask, vis_depth, vis_albedo