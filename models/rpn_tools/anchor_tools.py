
import torch

class Anchor(object):

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        h_ratios = torch.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)

        # yapf: disable
        base_anchors = torch.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            dim=-1).round()
        # yapf: enable

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors.to(device)

        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, all_anchors, im_info):
        total_anchors = all_anchors.size(0)
        _allowed_border = 0
        keep = ((all_anchors[:, 0] >= -_allowed_border) &
                (all_anchors[:, 1] >= -_allowed_border) &
                (all_anchors[:, 2] < int(im_info[1]) + _allowed_border) &
                (all_anchors[:, 3] < int(im_info[0]) + _allowed_border))

        inds_inside = torch.nonzero(keep).view(-1)

        return inds_inside


# class Anchor(object):
#
#     def __init__(self, stride, ratios, scales):
#
#         self.stride = stride
#         self.base_anchor = torch.from_numpy(generate_anchors(base_size=stride, scales=np.array(scales), ratios=np.array(ratios))).float()
#
#     def _meshgrid(self, x, y, row_major=True):
#         xx = x.repeat(len(y))
#         yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
#         if row_major:
#             return xx, yy
#         else:
#             return yy, xx
#
#     def __call__(self, feat_height, feat_width, device):
#         base_anchors = self.base_anchor.to(device)
#         shift_x = torch.arange(0, feat_width, device=device) * self.stride
#         shift_y = torch.arange(0, feat_height, device=device) * self.stride
#         shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
#         shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
#         shifts = shifts.type_as(base_anchors)
#
#         all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
#         all_anchors = all_anchors.view(-1, 4)
#
#         return all_anchors