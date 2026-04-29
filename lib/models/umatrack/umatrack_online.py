from functools import partial
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath, Mlp, trunc_normal_
from lib.utils.misc import is_main_process
from lib.models.umatrack.head_256 import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from lib.models.umatrack.pos_utils import get_2d_sincos_pos_embed
from lib.models.umatrack.efficientvit import EfficientViT_M4
from lib.models.umatrack.score_decoder import ScoreDecoder
class UnidirectionalMixedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_mem = None

        qkvs = []
        # dws = []
        # bns = []
        for i in range(num_heads):
            qkvs.append(nn.Linear(dim // (num_heads), (dim * 3) // (num_heads), bias=qkv_bias))
            # dws.append(torch.nn.Conv(dim // (num_heads), dim // (num_heads), 3, 1, 1, groups=dim // (num_heads)))
            # bn.append(torch.nn.BatchNorm2d)
        self.qkvs = torch.nn.ModuleList(qkvs)
        # self.dws = torch.nn.ModuleList(dws)
        self.x_t = None
    def forward(self, x_t, x_s, t_h, t_w, s_h, s_w):
        """
        x is a concatenated vector of template and search region features.
        only add search
        """
        x = torch.cat([x_t, x_s], dim=1)
        B, N, C = x.shape
        feats_in = x.chunk(len(self.qkvs), dim=2)  # 分块
        feats_ins = x_s.chunk(len(self.qkvs), dim=2)  # 分块
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0:  # add the previous output to the input
                feat_t, feat_so = torch.split(feats_in[i], [t_h * t_w *2, s_h * s_w], dim=1)
                feat_s = feat_s + feats_ins[i]
                feat = torch.cat([feat_t, feat_s], dim=1)
                # feat = feat + feats_in[i]

            # feat = qkv(feat)
            # xt, xot, xs = torch.split(x, [t_h * t_w, t_h * t_w, s_h * s_w], dim=2)

            qkvss = qkv(feat).reshape(B, N, 3, 1, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkvss.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

            q_mt, q_s = torch.split(q, [t_h * t_w * 2, s_h * s_w], dim=2)
            # k_mt, k_s = torch.split(k, [t_h * t_w * 2, s_h * s_w], dim=2)
            # v_mt, v_s = torch.split(v, [t_h * t_w * 2, s_h * s_w], dim=2)

            # asymmetric mixed attention
            #     attn = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
            #     attn = attn.softmax(dim=-1)
            #     attn = self.attn_drop(attn)
            #     x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, t_h*t_w*2, C // self.num_heads)

            attn = (q_s @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_s = (attn @ v).transpose(1, 2).reshape(B, s_h * s_w, C // self.num_heads)
            feat_s = x_s
            feats_out.append(feat_s)
        x = self.proj(torch.cat(feats_out, 2))
        x = self.proj_drop(x)
        return x

    def forward_test(self, x_s, s_h, s_w):
        """
        x is a concatenated vector of template and search region features.
        """

        t_h = self.t_h
        t_w = self.t_w

        x_t = self.x_t
        # x_s = x
        x = torch.cat([x_t, x_s], dim=1)
        B, N, C = x.shape
        feats_in = x.chunk(len(self.qkvs), dim=2)  # 分块
        feats_ins = x_s.chunk(len(self.qkvs), dim=2)  # 分块
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0:  # add the previous output to the input
                feat_t, feat_so = torch.split(feats_in[i], [t_h * t_w * 2, s_h * s_w], dim=1)
                feat_s = feat_s + feats_ins[i]
                feat = torch.cat([feat_t, feat_s], dim=1)
            qkvss = qkv(feat).reshape(B, N, 3, 1, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkvss.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
            q_mt, q_s = torch.split(q, [t_h * t_w * 2, s_h * s_w], dim=2)
            attn = (q_s @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_s = (attn @ v).transpose(1, 2).reshape(B, s_h * s_w, C // self.num_heads)
            feat_s = x_s
            feats_out.append(feat_s)
        x = self.proj(torch.cat(feats_out, 2))
        x = self.proj_drop(x)
        return x

    def set_online(self, x, t_h, t_w):
        """
        x is a concatenated vector of template and search region features.
        """
        self.x_t = x
        self.t_h = t_h
        self.t_w = t_w

        return x


class MHM(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = UnidirectionalMixedAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, t_h, t_w, s_h, s_w):
        x = self.norm1(x)
        # B, N, C = x.shape
        x_t, x_s = torch.split(x, [t_h * t_w * 2, s_h * s_w], dim=1)

        x = x_s + self.drop_path1(self.attn(x_t, x_s, t_h, t_w, s_h, s_w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

    def forward_test(self, x, s_h, s_w):
        x = self.norm1(x)
        # B, N, C = x.shape
        x = x + self.drop_path1(self.attn.forward_test(x, s_h, s_w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

    def set_online(self, x, t_h, t_w):
        x = self.attn.set_online(self.norm1(x), t_h, t_w)
        return x

class EfficietMHM(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size_s=288, img_size_t=128, patch_size=[4, 2, 2], embed_dim=[256, 384, 768],
                 depth=[2, 2, 11], num_heads=12, mlp_ratio=[4, 4, 4], in_chans=3, num_classes=1000,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed4 = nn.Linear(embed_dim[1], embed_dim[2])
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.blocks3 = nn.ModuleList([
            MHM(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + depth[1] + i], norm_layer=norm_layer)
            for i in range(depth[2])])

        self.apply(self._init_weights)

        self.grid_size_s = img_size_s // (patch_size[0] * patch_size[1] * patch_size[2])
        self.grid_size_t = img_size_t // (patch_size[0] * patch_size[1] * patch_size[2])
        self.num_patches_s = self.grid_size_s ** 2
        self.num_patches_t = self.grid_size_t ** 2
        self.pos_embed_s = nn.Parameter(torch.zeros(1, self.num_patches_s, embed_dim[2]), requires_grad=False)
        self.pos_embed_t = nn.Parameter(torch.zeros(1, self.num_patches_t, embed_dim[2]), requires_grad=False)

        self.init_pos_embed()
        self.efficientvit = EfficientViT_M4()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def init_pos_embed(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_t = get_2d_sincos_pos_embed(self.pos_embed_t.shape[-1], int(self.num_patches_t ** .5),
                                              cls_token=False)
        self.pos_embed_t.data.copy_(torch.from_numpy(pos_embed_t).float().unsqueeze(0))

        pos_embed_s = get_2d_sincos_pos_embed(self.pos_embed_s.shape[-1], int(self.num_patches_s ** .5),
                                              cls_token=False)
        self.pos_embed_s.data.copy_(torch.from_numpy(pos_embed_s).float().unsqueeze(0))

    def forward(self, x_t, x_ot, x_s):

        x_t = self.efficientvit(x_t)
        x_t = x_t.flatten(2).permute(0, 2, 1)
        x_t = self.patch_embed4(x_t)

        x_ot = self.efficientvit(x_ot)
        x_ot = x_ot.flatten(2).permute(0, 2, 1)
        x_ot = self.patch_embed4(x_ot)

        # st = time.time()
        x_s = self.efficientvit(x_s)
        x_s = x_s.flatten(2).permute(0, 2, 1)
        x_s = self.patch_embed4(x_s)

        B, C = x_t.size(0), x_t.size(-1)
        H_s = W_s = self.grid_size_s
        H_t = W_t = self.grid_size_t

        x_s = x_s + self.pos_embed_s
        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t

        x = torch.cat([x_t, x_ot, x_s], dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks3:
            x = blk(x, H_t, W_t, H_s, W_s)
        x_s = x
        x_t_2d = x_t.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_ot_2d = x_ot.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_s_2d = x_s.transpose(1, 2).reshape(B, C, H_s, W_s)

        return x_t_2d, x_ot_2d, x_s_2d

    def forward_test(self, x_s):
        x_s = self.efficientvit(x_s)
        x_s = x_s.flatten(2).permute(0, 2, 1)
        x_s = self.patch_embed4(x_s)
        H_s = W_s = self.grid_size_s
        x_s = x_s + self.pos_embed_s
        x_s = self.pos_drop(x_s)

        for blk in self.blocks3:
            x_s = blk.forward_test(x_s, H_s, W_s)
        x_s = rearrange(x_s, 'b (h w) c -> b c h w', h=H_s, w=H_s)

        return self.template, x_s

    def set_online(self, x_t, x_ot):
        ### conv embeddings for x_t
        x_t = self.efficientvit(x_t)
        x_t = x_t.flatten(2).permute(0, 2, 1)
        x_t = self.patch_embed4(x_t)

        x_ot = self.efficientvit(x_ot)
        x_ot = x_ot.flatten(2).permute(0, 2, 1)
        x_ot = self.patch_embed4(x_ot)

        # B, C = x_t.size(0), x_t.size(-1)
        H_t = W_t = self.grid_size_t

        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t

        x = torch.cat([x_t, x_ot], dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks3:
            x = blk.set_online(x, H_t, W_t)
        x_t = x[:, :H_t * W_t]
        x_t = rearrange(x_t, 'b (h w) c -> b c h w', h=H_t, w=W_t)

        self.template = x_t


def get_ef_mhm(config, train):
    img_size_s = config.DATA.SEARCH.SIZE
    img_size_t = config.DATA.TEMPLATE.SIZE
    ef = EfficietMHM(
    img_size_s=img_size_s, img_size_t=img_size_t, patch_size=[4, 2, 2], embed_dim=[128, 256, 256], depth=[0, 0, 1], num_heads=8, mlp_ratio=[4, 4, 4], qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6))

    if config.MODEL.BACKBONE.PRETRAINED and train:
        ckpt_path = config.MODEL.BACKBONE.PRETRAINED_PATH
        ckpt = torch.load(ckpt_path, map_location='cpu') #['model']
        new_dict = {}
        for k, v in ckpt.items():
            if 'pos_embed' not in k and 'mask_token' not in k:
                new_dict[k] = v
        missing_keys, unexpected_keys = vit.load_state_dict(new_dict, strict=False)
        if is_main_process():
            print("Load pretrained backbone checkpoint from:", ckpt_path)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained EfficientMHM done.")

    return ef


class UMATrackOnlineScore(nn.Module):
    def __init__(self, backbone, box_head, score_branch=None, head_type="CORNER", distill=False):
        """ Initializes the model.
        """
        super().__init__()
        self.backbone = backbone
        self.box_head = box_head
        self.score_branch = score_branch
        self.head_type = head_type
        self.distill = distill

    def forward(self, template, online_template, search, run_score_head=True, gt_bboxes=None, softmax=True):
        # search: (b, c, h, w)
        if template.dim() == 5:
            template = template.squeeze(0)
        if online_template.dim() == 5:
            online_template = online_template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)
        template, online_template, search = self.backbone(template, online_template, search)
        if self.distill:
            outputs_coord, prob_tl, prob_br = self.forward_head(search, template, run_score_head, gt_bboxes, softmax=softmax)
            return {"pred_boxes": outputs_coord, "prob_tl": prob_tl, "prob_br": prob_br}, None, None
        else:
            out, outputs_coord_new = self.forward_head(search, template, run_score_head, gt_bboxes)
            return out, outputs_coord_new

    def forward_test(self, search, run_score_head=True, gt_bboxes=None):
        if search.dim() == 5:
            search = search.squeeze(0)
        template, search = self.backbone.forward_test(search)
        out, outputs_coord_new = self.forward_head(search, template, run_score_head, gt_bboxes)

        return out, outputs_coord_new

    def set_online(self, template, online_template):
        if template.dim() == 5:
            template = template.squeeze(0)
        if online_template.dim() == 5:
            online_template = online_template.squeeze(0)
        self.backbone.set_online(template, online_template)

    def forward_head(self, search, template, run_score_head=True, gt_bboxes=None, softmax=True):
        """
        :param search: (b, c, h, w)
        :return:
        """
        out_dict = {}
        if self.distill:
            outputs_coord, prob_tl, prob_br = self.forward_box_head(search, softmax=softmax)
            return outputs_coord, prob_tl, prob_br
        else:
            out_dict_box, outputs_coord = self.forward_box_head(search)
            out_dict.update(out_dict_box)
            if run_score_head:
                # forward the classification head
                if gt_bboxes is None:
                    gt_bboxes = box_cxcywh_to_xyxy(outputs_coord.clone().view(-1, 4))
                # (b,c,h,w) --> (b,h,w)
                out_dict.update({'pred_scores': self.score_branch(search, template, gt_bboxes).view(-1)})
            return out_dict, outputs_coord

    def forward_box_head(self, search, softmax=True):
        """
        :param search: (b, c, h, w)
        :return:
        """
        if "CORNER" in self.head_type:
            # run the corner head
            b = search.size(0)

            if self.distill:
                coord_xyxy, prob_vec_tl, prob_vec_br = self.box_head(search, return_dist=True, softmax=softmax)
                outputs_coord = box_xyxy_to_cxcywh(coord_xyxy)
                outputs_coord_new = outputs_coord.view(b, 1, 4)
                return outputs_coord_new, prob_vec_tl, prob_vec_br
            else:
                outputs_coord = box_xyxy_to_cxcywh(self.box_head(search))
                outputs_coord_new = outputs_coord.view(b, 1, 4)
                out = {'pred_boxes': outputs_coord_new}
                return out, outputs_coord_new
        else:
            raise KeyError

def build_umatrack_online_score(cfg, settings=None, train=True) -> UMATrackOnlineScore:
    backbone = get_ef_mhm(cfg, train)
    score_branch = ScoreDecoder(pool_size=4, hidden_dim=cfg.MODEL.HIDDEN_DIM, num_heads=8)
    box_head = build_box_head(cfg)
    model = UMATrackOnlineScore(
        backbone,
        box_head,
        score_branch,
        head_type=cfg.MODEL.HEAD_TYPE
        # distill=cfg.TRAIN.DISTILL
    )
    if cfg.MODEL.PRETRAINED_STAGE1 and train:
        path = settings.stage1_model
        ckpt_path = path
        ckpt = torch.load(ckpt_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(ckpt['net'], strict=False)
        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)
        print("Loading pretrained mixformer weights done.")
    return model