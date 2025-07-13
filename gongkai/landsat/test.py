import os
import torch
import numpy as np
import PIL.Image as Image

from dat.change_detection import ChangeDetection_Landsat_nolabel
from models.net import ournet as Net
from utils.palette import color_map_Landsat_SCD as color_map
from utils.metric import IOUandSek

from tqdm import tqdm
from torch.utils.data import DataLoader
from thop import profile
import time
import argparse
import copy
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class DualOutput:
    def __init__(self, file_name):
        self.console = sys.stdout
        self.file = open(file_name, 'w')

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()

num_classes = 5



class Options:
    def __init__(self):
        parser = argparse.ArgumentParser('Semantic Change Detection')
        parser.add_argument("--data_name", type=str, default="Landsat")
        parser.add_argument("--Net_name", type=str, default="ournet")
        parser.add_argument("--lightweight", dest="lightweight", action="store_true",
                           help='lightweight head for fewer parameters and faster speed')
        parser.add_argument("--backbone", type=str, default="resnet34")
        parser.add_argument("--data_root", type=str, default=r"/home/user/zly/data3/Landsat/sample/t0.10/")
        parser.add_argument("--load_from", type=str,
                            default=r"/home/user/zly/data3/Landsat/pth/gongkai/t0.10/checkpoints/Landsat/Net/resnet34/.......pth")
        parser.add_argument("--test_batch_size", type=int, default=1)
        parser.add_argument("--pretrained", type=bool, default=True,
                           help='initialize the backbone with pretrained parameters')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        print(args)
        return args

def inference(args):
    
    working_path = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join('/home/user/zly/data3/Landsat/result/gongkai/t0.10/', 'pred_results', args.data_name, args.Net_name, args.backbone)
    pred_save_path1 = os.path.join(pred_dir, 'pred1')
    pred_save_path2 = os.path.join(pred_dir, 'pred2')
    pred_save_path1_rgb = os.path.join(pred_dir, 'pred1_rgb')
    pred_save_path2_rgb = os.path.join(pred_dir, 'pred2_rgb')
    pred_save_path1_semantic = os.path.join(pred_dir, 'pred1_semantic')
    pred_save_path2_semantic = os.path.join(pred_dir, 'pred2_semantic')
    pred_save_path3 = os.path.join(pred_dir, 'pred_change')

    if not os.path.exists(pred_save_path1): os.makedirs(pred_save_path1)
    if not os.path.exists(pred_save_path2): os.makedirs(pred_save_path2)
    if not os.path.exists(pred_save_path1_rgb): os.makedirs(pred_save_path1_rgb)
    if not os.path.exists(pred_save_path2_rgb): os.makedirs(pred_save_path2_rgb)
    if not os.path.exists(pred_save_path1_semantic): os.makedirs(pred_save_path1_semantic)
    if not os.path.exists(pred_save_path2_semantic): os.makedirs(pred_save_path2_semantic)
    if not os.path.exists(pred_save_path3): os.makedirs(pred_save_path3)
    log_file = '/home/user/zly/data3/Landsat/result/gongkai/t0.10/test_t0.10.txt'  
    sys.stdout = DualOutput(log_file)
    testset = ChangeDetection_Landsat_nolabel(root=args.data_root,mode="test")
    testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                                pin_memory=True, num_workers=0, drop_last=False)
    model = Net(args.backbone, args.pretrained, len(ChangeDetection_Landsat_nolabel.CLASSES)-1,)

    if args.load_from:
        model.load_state_dict(torch.load(args.load_from), strict=True)

    model = model.cuda()
    model.eval()

    tbar = tqdm(testloader)
    metric = IOUandSek(num_classes=len(ChangeDetection_Landsat_nolabel.CLASSES))
    begin_time = time.time()
    with torch.no_grad():
        for img1, img2, label1, label2, label_bn, class_names_1, confidences_1, class_names_2, confidences_2, id  in tbar:
            img1, img2 = img1.cuda(), img2.cuda()

            label1 = (label1 / 10).long()
            label2 = (label2 / 10).long()

            out1, out2, out_bn, _,  = model(img1, img2,  confidences_1, confidences_2)
            pred1_seg = torch.argmax(out1, dim=1).cpu().numpy() + 1
            pred2_seg = torch.argmax(out2, dim=1).cpu().numpy() + 1

            out_bn11 = torch.sigmoid(out_bn)
            out_bn = (out_bn > (1 - out_bn11 * 0.05).detach()).cpu().numpy().astype(np.uint8)
            out1 = copy.deepcopy(pred1_seg)
            out2 = copy.deepcopy(pred2_seg)

            out1[out_bn == 0] = 0
            out2[out_bn == 0] = 0

            cmap = color_map()

            for i in range(out1.shape[0]):
                mask1 = Image.fromarray(out1[i].astype(np.uint8))
                mask1.save(os.path.join(pred_save_path1, id[i]))
                mask1_rgb = mask1.convert('P')
                mask1_rgb.putpalette(cmap)
                mask1_rgb.save(os.path.join(pred_save_path1_rgb, id[i]))
                mask1_seg = Image.fromarray(pred1_seg[i].astype(np.uint8)).convert('P')
                mask1_seg.putpalette(cmap)
                mask1_seg.save(os.path.join(pred_save_path1_semantic, id[i]))

                mask2 = Image.fromarray(out2[i].astype(np.uint8))
                mask2.save(os.path.join(pred_save_path2, id[i]))
                mask2_rgb = mask2.convert('P')
                mask2_rgb.putpalette(cmap)
                mask2_rgb.save(os.path.join(pred_save_path2_rgb, id[i]))
                mask2_seg = Image.fromarray(pred2_seg[i].astype(np.uint8)).convert('P')
                mask2_seg.putpalette(cmap)
                mask2_seg.save(os.path.join(pred_save_path2_semantic, id[i]))

                mask_bn = Image.fromarray(out_bn[i]*255)
                mask_bn.save(os.path.join(pred_save_path3, id[i]))

            metric.add_batch(out1, label1.numpy())
            metric.add_batch(out2, label2.numpy())


        change_ratio, OA, mIoU, Sek, Fscd, Score, Precision_scd, Recall_scd = metric.evaluate_inference()

        print('==>change_ratio', change_ratio)
        print('==>oa', OA)
        print('==>miou', mIoU)
        print('==>sek', Sek)
        print('==>Fscd', Fscd)
        print('==>score', Score)
        print('==>SC_Precision', Precision_scd)
        print('==>SC_Recall', Recall_scd)

        time_use = time.time() - begin_time

    metric_file = os.path.join(pred_dir, 'metric.txt')
    f = open(metric_file, 'w', encoding='utf-8')
    f.write("Data：" + str(args.data_name) + '\n')
    f.write("model：" + str(args.Net_name) + '\n')
    f.write("##################### metric #####################"+'\n')
    f.write("infer time (s) ：" + str(round(time_use, 2)) + '\n')
    f.write('\n')
    f.write("change_ratio (%) ：" + str(round(change_ratio * 100, 2)) + '\n')
    f.write("OA (%) ：" + str(round(OA * 100, 2)) + '\n')
    f.write("mIoU (%) ：" + str(round(mIoU * 100, 2)) + '\n')
    f.write("Sek (%) ：" + str(round(Sek * 100, 2)) + '\n')
    f.write("Fscd (%) ：" + str(round(Fscd * 100, 2)) + '\n')
    f.write("Score (%) ：" + str(round(Score * 100, 2)) + '\n')
    f.write("Precision_scd (%) ：" + str(round(Precision_scd * 100, 2)) + '\n')
    f.write("Recall_scd (%) ：" + str(round(Recall_scd * 100, 2)) + '\n')

    f.close()


if __name__ == "__main__":
    args = Options().parse()
    inference(args)