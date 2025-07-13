from torch.nn import CrossEntropyLoss, BCELoss
from models.Backbones.resnet import resnet18, resnet34, resnet50
from models.Decoders.Decoder4 import  Seg_Decoder_ResNet, CD_Decoder_ResNet
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
from skimage import exposure

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def get_backbone(backbone, pretrained):
    if backbone == 'resnet34':
        backbone = resnet34(pretrained)
    else:
        exit("\nError: BACKBONE \'%s\' is not implemented!\n" % backbone)
    return backbone


class MinusConv(nn.Module):

    def __init__(self, in_channels):
        super(MinusConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

def compute_loss(feature_maps, feature_vectors):
    loss = 0
    for fmap, fvec in zip(feature_maps, feature_vectors):

        fmap_gapped = F.adaptive_avg_pool2d(fmap, (1, 1)).squeeze(-1).squeeze(-1)

        fmap_normalized = F.normalize(fmap_gapped, p=2, dim=-1)
        fvec_normalized = F.normalize(fvec, p=2, dim=-1)

        loss += F.mse_loss(fmap_normalized, fvec_normalized)

    return loss


class ChannelAttention_text(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention_text, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(56, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, confidences):
        attn = self.fc(confidences)
        return attn.unsqueeze(2).unsqueeze(3)


class SpatialAttention_text(nn.Module):

    def __init__(self, in_channels):
        super(SpatialAttention_text, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.conv(x))
        return attn


class ExtractFeat(nn.Module):
    def __init__(self):
        super(ExtractFeat, self).__init__()



        self.ca1 = ChannelAttention_text(64)
        self.ca2 = ChannelAttention_text(128)
        self.ca3 = ChannelAttention_text(256)
        self.ca4 = ChannelAttention_text(512)
        self.ca5 = ChannelAttention_text(512)


        self.sa1 = SpatialAttention_text(64)
        self.sa2 = SpatialAttention_text(128)
        self.sa3 = SpatialAttention_text(256)
        self.sa4 = SpatialAttention_text(512)
        self.sa5 = SpatialAttention_text(512)

    def forward(self, feature1_t, confidences_1):
        features = []
        text_features = []

        for i, feature in enumerate(feature1_t):
            batch_size, channels, height, width = feature.shape

            confidences = torch.stack(confidences_1, dim=0).to(dtype=torch.float32, device=feature.device).T

            if i == 0:

                ca = self.ca1(confidences)
                sa = self.sa1(feature)
            elif i == 1:

                ca = self.ca2(confidences)
                sa = self.sa2(feature)
            elif i == 2:

                ca = self.ca3(confidences)
                sa = self.sa3(feature)
            elif i == 3:

                ca = self.ca4(confidences)
                sa = self.sa4(feature)
            elif i == 4:

                ca = self.ca5(confidences)
                sa = self.sa5(feature)


            feature = feature * ca


            feature = feature * sa


            feature = feature + feature1_t[i]

            text_features.append(ca.squeeze(2).squeeze(2))
            features.append(feature)

        return features, text_features


class SimpleTemporalAttention(nn.Module):
    def __init__(self, in_dim):
        super(SimpleTemporalAttention, self).__init__()
        self.conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, input1, input2, input12_diff):

        mean1 = torch.mean(input1, dim=(2, 3), keepdim=True)
        mean2 = torch.mean(input2, dim=(2, 3), keepdim=True)

        weight = self.conv(mean1 + mean2)
        weight = torch.sigmoid(weight)

        out = self.gamma * weight * input12_diff + input12_diff  # 残差连接
        return out


class ournet(nn.Module):
    def __init__(self, backbone, pretrained, nclass,):
        super(ournet, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.backbone = get_backbone(backbone, pretrained)
        self.backbone_class = get_backbone(backbone, pretrained)

        if backbone == "resnet34":
            self.channel_nums = [64, 128, 256, 512]

        if backbone == "resnet34":
            self.Seg_Decoder = Seg_Decoder_ResNet(self.channel_nums)
            self.CD_Decoder = CD_Decoder_ResNet(self.channel_nums)

        self.extract_feat = ExtractFeat()

        self.adjust_conv = nn.ModuleList([
            nn.Conv2d(in_channels=feature_orig_channels, out_channels=feature_diff_channels, kernel_size=1)
            for feature_orig_channels, feature_diff_channels in zip(
                [320, 640, 1280, 2560, 2560],
                [64, 128, 256, 512, 512]
            )
        ])

        self.sta = SimpleTemporalAttention(256)

        self.contexts2 = nn.Parameter(torch.randn(1, 1, 56, 512))

        self.minus_conv = nn.ModuleList([
            MinusConv(64),
            MinusConv(128),
            MinusConv(256),
            MinusConv(512),
            MinusConv(512)
        ])


        self.seg_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.2, False),
            nn.Conv2d(128, self.nclass, 1, bias=True)
        )

        self.resCD_conv = nn.Conv2d(64, 256, kernel_size=1, stride=1)


        self.classifierCD = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(),
                                          nn.Conv2d(128, 1, kernel_size=1))


    def forward(self, x1, x2,  confidences_1, confidences_2): #
        b, c, h, w = x1.shape

        feature1 = self.backbone.forward(x1)
        feature2 = self.backbone.forward(x2)


        feature1_t = tuple(feature1)
        feature2_t = tuple(feature2)


        feature1_fea_text= self.extract_feat(feature1_t,  confidences_1)
        feature2_fea_text = self.extract_feat(feature2_t,  confidences_2)
        feature1 = feature1_fea_text[0]
        feature2 = feature2_fea_text[0]
        feature1_text = feature1_fea_text[1]
        feature2_text = feature2_fea_text[1]


        feature_orig = [torch.cat([feature1[i], feature2[i]], dim=1) for i in range(len(feature1))]  # 拼接特征

        feature_minus = [(torch.abs(feature1[i] - feature2[i])) for i in
                         range(len(feature1))]

        feature_diff = [
            F.sigmoid(1 - torch.cosine_similarity(feature1[i], feature2[i], dim=1)).unsqueeze(1)
            for i in range(len(feature1))
        ]

        feature_text_diff = []
        for i in range(len(feature1_text)):
            feature_text_diff.append(torch.abs(feature1_text[i] - feature2_text[i]))


        feature_fusion = [
            self.adjust_conv[i](
                torch.cat([feature_orig[i]  * feature_diff[i], feature_minus[i], feature_orig[i]], dim=1))
            for i in range(len(feature_orig))
        ]



        loss1 = compute_loss(feature_fusion, feature_text_diff)

        feature1_he = feature1
        feature2_he = feature2
        feature_diff_he = feature_fusion


        out1 = self.Seg_Decoder(feature1_he)
        out2 = self.Seg_Decoder(feature2_he)
        out_diff = self.CD_Decoder(feature_diff_he)


        out_diff = self.sta(out1, out2, out_diff) 

        change = F.interpolate(self.classifierCD(out_diff), size=(h, w), mode='bilinear', align_corners=False)
        seg1 = F.interpolate(self.seg_conv(out1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out2), size=(h, w), mode='bilinear', align_corners=False)


        change1 =  torch.abs(seg1[:, 0, :, :] - seg2[:, 0, :, :]) + torch.abs(
                    seg1[:, 1, :, :] - seg2[:, 1, :, :]) + torch.abs(seg1[:, 2, :, :] - seg2[:, 2, :, :]) + torch.abs(
                    seg1[:, 3, :, :] - seg2[:, 3, :, :]) + torch.abs(seg1[:, 4, :, :] - seg2[:, 4, :, :]) + torch.abs(
                    seg1[:, 5, :, :] - seg2[:, 5, :, :])

        loss2 = F.mse_loss(torch.sigmoid(change1), torch.sigmoid(change.squeeze(1)))

        loss_laycha = loss1 + loss2



        return seg1, seg2, change.squeeze(1), loss_laycha,


if __name__ == '__main__':
    ()