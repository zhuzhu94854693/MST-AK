from itertools import cycle
from dat.change_detection import ChangeDetection_SECOND_label, ChangeDetection_SECOND_nolabel
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from models.net import ournet as Net
from utils.palette import color_map
from utils.metric import IOUandSek
from utils.loss import ChangeSimilarity, DiceLoss, ContrastiveLoss
import os
import torch.nn as nn
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import Adam, AdamW, SGD, lr_scheduler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
working_path = os.path.dirname(os.path.abspath(__file__))

import itertools

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True, warn_only=True)


num_classes = 7
ST_COLORMAP = [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0], [128, 0, 0], [255, 0, 0]]
ST_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']

MEAN_A = np.array([113.40, 114.08, 116.45])
STD_A = np.array([48.30, 46.27, 48.14])
MEAN_B = np.array([111.07, 114.04, 118.18])
STD_B = np.array([49.41, 47.01, 47.94])

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i



def Colorls2Index(ColorLabels):
    IndexLabels = []
    for i, data in enumerate(ColorLabels):
        IndexMap = Color2Index(data)
        IndexLabels.append(IndexMap)
    return IndexLabels


def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]

    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap


def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]




def batch_color_to_index(tensor):

    batch_size, height, width, channels = tensor.shape

    index_tensor = torch.zeros((batch_size, height, width), dtype=torch.int32)

    for i in range(batch_size):

        color_image = tensor[i].cpu().numpy().astype(np.int32)

        index_map = Color2Index(color_image)

        index_tensor[i] = torch.from_numpy(index_map)

    return index_tensor


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser('Semantic Change Detection')
        parser.add_argument("--data_name", type=str, default=r"SECOND")
        parser.add_argument("--Net_name", type=str, default="Net")
        parser.add_argument("--backbone", type=str, default="resnet34")
        parser.add_argument("--data_root", type=str, default=r'/home/user/zly/data3/SECOND/sample/t0.10/')
        parser.add_argument("--log_dir", type=str)
        parser.add_argument("--batch_size", type=int, default=6)
        parser.add_argument("--val_batch_size", type=int, default=8)
        parser.add_argument("--test_batch_size", type=int, default=1)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--lr", type=float, default=0.0003)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--pretrain_from", type=str, help='train from a checkpoint')
        parser.add_argument("--load_from", type=str, help='load trained model to generate predictions of validation set')
        parser.add_argument("--pretrained", type=bool, default=True, help='initialize the backbone with pretrained parameters')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        print(args)
        return args


def update_ema_variables(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


class Trainer:
    def __init__(self, args):
        args.log_dir = os.path.join('/home/user/zly/data3/SECOND/pth/gongkai/', 'logs_t0.10', args.data_name,
                                    args.Net_name, args.backbone)
        if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
        self.writer = SummaryWriter(args.log_dir)
        self.args = args

        trainset_label = ChangeDetection_SECOND_label(root=args.data_root, mode="train")
        trainset_nolabel = ChangeDetection_SECOND_nolabel(root=args.data_root,mode="train")
        valset = ChangeDetection_SECOND_nolabel(root=args.data_root,mode="val")
        testset = ChangeDetection_SECOND_nolabel(root=args.data_root,mode="test")
        self.trainloader_label = DataLoader(trainset_label, batch_size=args.batch_size, shuffle=True,
                                            pin_memory=False, num_workers=6, drop_last=True)
        self.trainloader_nolabel = DataLoader(trainset_nolabel, batch_size=args.batch_size, shuffle=True,
                                              pin_memory=False, num_workers=6, drop_last=True)
        self.valloader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=False,
                                    pin_memory=True, num_workers=6, drop_last=False)
        self.model = Net(args.backbone, args.pretrained, len(trainset_label.CLASSES) - 1,)

        self.ema_model = Net(args.backbone, args.pretrained, len(trainset_label.CLASSES) - 1,)
        if args.pretrain_from:
            self.model.load_state_dict(torch.load(args.pretrain_from), strict=False)
            self.ema_model.load_state_dict(torch.load(args.pretrain_from), strict=False)
        if args.load_from:
            self.model.load_state_dict(torch.load(args.load_from), strict=True)
            self.ema_model.load_state_dict(torch.load(args.load_from), strict=True)


        self.criterion_contrastive2 = ContrastiveLoss()
        self.criterion_seg = CrossEntropyLoss(ignore_index=-1)
        self.criterion_bn_2 = DiceLoss()
        self.criterion_sc = ChangeSimilarity()
        self.criterion_bn = nn.BCEWithLogitsLoss(reduction='none').cuda()

        self.optimizer = AdamW([{"params": [param for name, param in self.model.named_parameters()
                                            if "backbone" in name], "lr": args.lr},
                                {"params": [param for name, param in self.model.named_parameters()
                                            if "backbone" not in name], "lr": args.lr * 1}],
                               lr=args.lr, weight_decay=args.weight_decay)
        self.model = self.model.cuda()
        self.ema_model = self.ema_model.cuda()
        self.iters = 0
        self.total_iters = len(self.trainloader_nolabel) * args.epochs
        self.previous_best = 0.0
        self.seg_best = 0.0
        self.change_best = 0.0

    def training(self, epoch):
        curr_epoch = epoch
        self.model.train()
        self.ema_model.train()
        total_loss = 0.0
        total_loss_seg = 0.0
        total_loss_bn = 0.0
        total_loss_similarity = 0.0

        if curr_epoch < 10:

            tbar = tqdm(self.trainloader_label,total=len(self.trainloader_label))
            curr_iter = curr_epoch * len(self.trainloader_label)
            for i, (img1_label, img2_label, mask1_label, mask2_label, mask_bn_label, class_names_1_label, confidences_1_label,
            class_names_2_label, confidences_2_label, _) in enumerate(tbar):
                running_iter = curr_iter + i + 1
                img1_label, img2_label = img1_label.cuda(), img2_label.cuda()
                mask1_label, mask2_label,  = mask1_label.cuda(), mask2_label.cuda(),
                mask1_label = batch_color_to_index(mask1_label).cuda().long()
                mask2_label = batch_color_to_index(mask2_label).cuda().long()

                confidences_1_label = [tensor.to("cuda") for tensor in confidences_1_label]
                confidences_2_label = [tensor.to("cuda") for tensor in confidences_2_label]

                mask_bn_label = (mask1_label > 0).cuda().float()

                out1, out2, out_bn, kk,  = self.model(img1_label, img2_label,  confidences_1_label, confidences_2_label)

                loss1 = self.criterion_seg(out1, mask1_label - 1)
                loss2 = self.criterion_seg(out2, mask2_label - 1)
                loss_seg = loss1 * 0.5 + loss2 * 0.5

                loss_similarity = self.criterion_sc(out1[:, 0:], out2[:, 0:], mask_bn_label)

                loss_bn_1 = self.criterion_bn(out_bn, mask_bn_label)
                loss_bn_1[mask_bn_label == 1] *= 2
                loss_bn_1 = loss_bn_1.mean()
                loss_bn_2 = self.criterion_bn_2(out_bn, mask_bn_label)
                loss_bn = loss_bn_1 + loss_bn_2
                loss = loss_bn + loss_seg + loss_similarity + kk

                total_loss_seg += loss_seg.item()
                total_loss_similarity += loss_similarity.item()
                total_loss_bn += loss_bn.item()
                total_loss += loss.item()

                self.iters += 1
                lr = self.args.lr * (1. - float(self.iters) / self.total_iters) ** 1.5
                self.optimizer.param_groups[0]["lr"] = lr
                self.optimizer.param_groups[1]["lr"] = lr

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    update_ema_variables(self.model, self.ema_model, alpha=0.99)

                tbar.set_description("Loss: %.3f, Semantic Loss: %.3f, Binary Loss: %.3f, Similarity Loss: %.3f" %
                                     (total_loss / (i + 1), total_loss_seg / (i + 1), total_loss_bn / (i + 1),
                                      total_loss_similarity / (i + 1)))

                self.writer.add_scalar('train total_loss', total_loss / (i + 1), running_iter)
                self.writer.add_scalar('train seg_loss', total_loss_seg / (i + 1), running_iter)
                self.writer.add_scalar('train bn_loss', total_loss_bn / (i + 1), running_iter)
                self.writer.add_scalar('train sc_loss', total_loss_similarity / (i + 1), running_iter)
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]["lr"], running_iter)

        else:

            tbar = tqdm(zip((self.trainloader_label), cycle(self.trainloader_nolabel)),total=len(self.trainloader_nolabel))
            curr_iter = curr_epoch * len(self.trainloader_nolabel)
            for i, ((img1_label, img2_label, mask1_label, mask2_label, mask_bn_label, class_names_1_label,
                     confidences_1_label, class_names_2_label, confidences_2_label,_),
                    (img1_nolabel, img2_nolabel, mask1_nolabel, mask2_nolabel, mask_bn_nolabel, class_names_1_nolabel,
                     confidences_1_nolabel, class_names_2_nolabel, confidences_2_nolabel,_)) in enumerate(tbar):
                running_iter = curr_iter + i + 1
                img1_label, img2_label = img1_label.cuda(), img2_label.cuda()
                mask1_label, mask2_label = mask1_label.cuda(), mask2_label.cuda()
                mask1_label = batch_color_to_index(mask1_label).cuda().long()
                mask2_label = batch_color_to_index(mask2_label).cuda().long()

                mask_bn_label = (mask1_label > 0).cuda().float()


                out1, out2, out_bn, pp,  = self.model(img1_label, img2_label, confidences_1_label, confidences_2_label)



                loss1 = self.criterion_seg(out1, mask1_label - 1)
                loss2 = self.criterion_seg(out2, mask2_label - 1)
                loss_seg = loss1 * 0.5 + loss2 * 0.5

                loss_similarity = self.criterion_sc(out1[:, 0:], out2[:, 0:], mask_bn_label)

                loss_bn_1 = self.criterion_bn(out_bn, mask_bn_label)
                loss_bn_1[mask_bn_label == 1] *= 2
                loss_bn_1 = loss_bn_1.mean()
                loss_bn_2 = self.criterion_bn_2(out_bn, mask_bn_label)
                loss_bn = loss_bn_1 + loss_bn_2
                loss = loss_bn + loss_seg + loss_similarity + pp

                out1_nolabel, out2_nolabel, out_bn_nolabel, out11_nolabel, = self.model( img1_nolabel.cuda(), img2_nolabel.cuda(), confidences_1_nolabel, confidences_2_nolabel)


                with torch.no_grad():
                    pseudo_out1, pseudo_out2, pseudo_out_bn, pseudo_out11, = self.ema_model(img1_nolabel.cuda(), img2_nolabel.cuda(), confidences_1_nolabel, confidences_2_nolabel)

                pseudo_out11 = torch.argmax(pseudo_out1, dim=1)
                pseudo_out22 = torch.argmax(pseudo_out2, dim=1)
                pseudo_out_bn = (pseudo_out_bn > 0.5).float()


                pse_loss1 = self.criterion_seg(out1_nolabel, pseudo_out11)
                pse_loss2 = self.criterion_seg(out2_nolabel, pseudo_out22)
                pse_loss_seg = pse_loss1 * 0.5 + pse_loss2 * 0.5
                pse_loss_similarity = self.criterion_sc(out1_nolabel[:, 0:], out2_nolabel[:, 0:], pseudo_out_bn)
                pse_loss_bn_1 = self.criterion_bn(out_bn_nolabel, pseudo_out_bn)
                pse_loss_bn_1[pseudo_out_bn == 1] *= 2
                pse_loss_bn_1 = pse_loss_bn_1.mean()
                pse_loss_bn_2 = self.criterion_bn_2(out_bn_nolabel, pseudo_out_bn)
                pse_loss_bn = pse_loss_bn_1 + pse_loss_bn_2

                loss = loss + 0.2 * (pse_loss_bn + pse_loss_similarity + pse_loss_seg)  # + contrastive_loss2)

                total_loss_seg += loss_seg.item()
                total_loss_similarity += loss_similarity.item()
                total_loss_bn += loss_bn.item()
                total_loss += loss.item()

                self.iters += 1
                lr = self.args.lr * (1. - float(self.iters) / self.total_iters) ** 1.5
                self.optimizer.param_groups[0]["lr"] = lr.item() if isinstance(lr, torch.Tensor) else lr
                self.optimizer.param_groups[1]["lr"] = lr.item() if isinstance(lr, torch.Tensor) else lr


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    update_ema_variables(self.model, self.ema_model, alpha=0.99)

                tbar.set_description("Loss: %.3f, Semantic Loss: %.3f, Binary Loss: %.3f, Similarity Loss: %.3f" %
                                     (total_loss / (i + 1), total_loss_seg / (i + 1), total_loss_bn / (i + 1),
                                      total_loss_similarity / (i + 1)))

                self.writer.add_scalar('train total_loss', total_loss / (i + 1), running_iter)
                self.writer.add_scalar('train seg_loss', total_loss_seg / (i + 1), running_iter)
                self.writer.add_scalar('train bn_loss', total_loss_bn / (i + 1), running_iter)
                self.writer.add_scalar('train sc_loss', total_loss_similarity / (i + 1), running_iter)
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]["lr"], running_iter)



    def validation(self, epoch):
        curr_epoch = epoch
        tbar = tqdm(self.valloader)
        self.model.eval()
        metric = IOUandSek(num_classes=len(ChangeDetection_SECOND_nolabel.CLASSES))

        with torch.no_grad():
            for img1, img2, mask1, mask2, mask_bn, class_names_1, confidences_1, class_names_2, confidences_2,_ in tbar:
                img1, img2 = img1.cuda(), img2.cuda()
                mask1 = batch_color_to_index(mask1).long()
                mask2 = batch_color_to_index(mask2).long()

                confidences_1 = [tensor.to("cuda") for tensor in confidences_1]
                confidences_2 = [tensor.to("cuda") for tensor in confidences_2]

                mask_bn = (mask1 > 0).float()

                out1, out2, out_bn, _,  = self.model(img1, img2, confidences_1, confidences_2)

                out1 = torch.argmax(out1, dim=1).cpu().numpy() + 1
                out2 = torch.argmax(out2, dim=1).cpu().numpy() + 1

                out_bn = (out_bn > (1 - out_bn * 1).detach()).cpu().numpy().astype(np.uint8)
                out1[out_bn == 0] = 0
                out2[out_bn == 0] = 0

                metric.add_batch(out1, mask1.numpy())
                metric.add_batch(out2, mask2.numpy())
                score, miou, sek, Fscd, OA, SC_Precision, SC_Recall = metric.evaluate_SECOND()

                tbar.set_description(
                    "miou: %.4f, sek: %.4f, score: %.4f, Fscd: %.4f, OA: %.4f, SC_Precision: %.4f, SC_Recall: %.4f" % (
                        miou, sek, score, Fscd, OA, SC_Precision, SC_Recall))

        if score >= self.previous_best:
            model_path = "/home/user/zly/data3/SECOND/pth/gongkai/t0.10/checkpoints/%s/%s/%s" % \
                         (self.args.data_name, self.args.Net_name, self.args.backbone)
            if not os.path.exists(model_path): os.makedirs(model_path)
            torch.save(self.model.state_dict(),
                       "/home/user/zly/data3/SECOND/pth/gongkai/t0.10/checkpoints/%s/%s/%s/epoch%i_Score%.2f_mIOU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth" %
                       (
                           self.args.data_name, self.args.Net_name, self.args.backbone, curr_epoch, score * 100,
                           miou * 100,
                           sek * 100, Fscd * 100, OA * 100))

            self.previous_best = score

        self.writer.add_scalar('val_Score', score, curr_epoch)
        self.writer.add_scalar('val_mIOU', miou, curr_epoch)
        self.writer.add_scalar('val_Sek', sek, curr_epoch)
        self.writer.add_scalar('val_Fscd', Fscd, curr_epoch)
        self.writer.add_scalar('val_OA', OA, curr_epoch)


if __name__ == "__main__":
    args = Options().parse()
    trainer = Trainer(args)

    if args.load_from:
        trainer.validation()

    for epoch in range(args.epochs):
        print("\n==> Epoches %i, learning rate = %.5f\t\t\t\t previous best = %.5f" %
              (epoch, trainer.optimizer.param_groups[0]["lr"], trainer.previous_best))


        trainer.training(epoch)
        trainer.validation(epoch)

