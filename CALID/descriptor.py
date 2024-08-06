from tqdm import tqdm
import argparse
from torchvision import models
import numpy as np
import torch
import PIL
from torchvision import transforms as T
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

class Feature_Extraction():
    def __init__(self, args):
        self.gpu = args.gpu
        self.net = args.net
        self.para = args.parameter
        self.mask_path = args.mask_path
        self.save_path = args.save_path
        self.ref_path = args.dataset_path
        self.pooling = args.pooling
        self.layer = args.layer
        self.box_path = args.box_path
        self.gpu = args.gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
        self.device = torch.device("cuda:" + self.gpu)
        self.model = self.load_model()
        self.features = {}

    def load_model(self):
        if self.net == "resnet50":
            if self.para == "pretrained":
                print("Loading Pretrain Resnet50......")
                model = models.resnet50(pretrained=True)
            elif self.para == "swav":
                print("Loading Swav Resnet50......")
                model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
            else:
                raise ValueError("Input unknown net parameter!")
        elif self.net == "resnet101":
            model = models.resnet101(pretrained=True)
        else:
            model = None
            raise ValueError("Input unknown net architecture!")
        model.eval().to(self.device)

        def hook(layer_name):
            def fn(module, input, output):
                self.features[layer_name] = output
            return fn

        handles = []
        for name, module in model.named_modules():
            if name == self.layer:
                handle = module.register_forward_hook(hook(name))
                handles.append(handle)
        return model
    

    def extract_ref(self):
        file_name = self.box_path
        file_names, file_boxes = get_bboxes_from_file(file_name)

        folder_path = self.save_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        ftr_save_path = os.path.join(folder_path, 'ref-ftr.txt')
        box_save_path = os.path.join(folder_path, 'ref-box.txt')
        f_ftr = open(ftr_save_path, 'wb')
        f_box = open(box_save_path, 'w')
    
        file_name = self.box_path
        frd = open(self.ref_path, 'r')
        lines = frd.readlines()
        
        with torch.no_grad():
            for idx, item in enumerate(lines):
                raw_img = PIL.Image.open(item.strip()).convert('RGB')
                pic_name = item.strip().split("/")[-1]
                w_resized, h_resized, rate = raw_img.size[0], raw_img.size[1], 1
                img_trans = T.Compose([
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                input_var = img_trans(raw_img).unsqueeze(0).to(self.device)
                output = self.model(input_var)
                boxes = np.array(file_boxes[idx])

                features, boxes_list = self.method_roi(self.layer, boxes, rate, self.pooling, pic_name, raw_img)
                save_bi_np_matrix(features,f_ftr)
                save_str_list(boxes_list, f_box)
            f_ftr.close()
            f_box.close()
    
    def extract_qry(self):
        file_name = self.box_path
        file_names, file_boxes = get_bboxes_from_file(file_name)

        folder_path = self.save_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        ftr_save_path = os.path.join(folder_path, 'qry-ftr.txt')
        box_save_path = os.path.join(folder_path, 'qry-box.txt')
        f_ftr = open(ftr_save_path, 'wb')
        f_box = open(box_save_path, 'w')
    
        file_name = self.box_path
        frd = open(self.box_path, 'r')
        lines = frd.readlines()
        
        with torch.no_grad():
            for idx, item in enumerate(lines):
                raw_img = PIL.Image.open(item.strip().split(">")[0]).convert('RGB')
                pic_name = item.strip().split(">")[0].split("/")[-1]
                w_resized, h_resized, rate = raw_img.size[0], raw_img.size[1], 1
                img_trans = T.Compose([
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                input_var = img_trans(raw_img).unsqueeze(0).to(self.device)
                output = self.model(input_var)
                boxes = np.array(file_boxes[idx])

                features, boxes_list = self.method_roi(self.layer, boxes, rate, self.pooling, pic_name, raw_img)
                save_bi_np_matrix(features,f_ftr)
                save_str_list(boxes_list, f_box)
            f_ftr.close()
            f_box.close()
        return 0
    
    def method_roi(self, layer, boxes, rate, pooling, pic_name, raw_img):
        feature_map = self.features[layer]
        if '4' in layer:
            self.stride = 32
        elif '3' in layer:
            self.stride = 16
        else:
            raise ValueError('Unknown layers input!')
        features = []
        box_lines = []
        for box_idx, box in enumerate(boxes):
            # resize box
            ori_box = boxes[box_idx]
            box = np.array([np.floor(ori_box[0] / rate / float(self.stride)),
                            np.floor(ori_box[1] / rate / float(self.stride)),
                            np.ceil(ori_box[2] / rate / float(self.stride)),
                            np.ceil(ori_box[3] / rate / float(self.stride))])
            box = map(int, box)
            xmin, ymin, xmax, ymax = box
            if xmin == xmax or ymin == ymax:
                continue

            if xmin > feature_map.shape[3] or ymin > feature_map.shape[2]:
                print(ori_box)
                print(rate)
                print(xmin, ymin, xmax, ymax)
                print(feature_map.shape)
                print(f'Wrong annotation: {pic_name}>{box_idx}')
                continue
            xmin = max(0, xmin)
            ymin = max(0, ymin)

            if pooling == 'mean':
                feature = torch.mean(feature_map[:, :, ymin:ymax, xmin:xmax], dim=[2, 3]).reshape(-1).cpu().numpy()
            elif pooling == 'max':
                feature, _ = torch.max(feature_map[:, :, ymin:ymax, xmin:xmax], dim=-2, keepdim=True)
                feature, _ = torch.max(feature, dim=-1)
                feature = feature.reshape(-1).cpu().numpy()
            elif pooling == "gem":
                fea_pow = torch.pow(feature_map[:, :, ymin:ymax, xmin:xmax], 3)
                fea_mean = torch.mean(fea_pow, dim=[2, 3]).reshape(-1)
                feature = torch.pow(fea_mean, 1/3).cpu().numpy()
            else:
                raise ValueError('Wrong pooling type!')

            features.append(feature)

            box_line = pic_name + ' ' + ' '.join(np.asarray(ori_box).astype(str)) + '\n'
            box_lines.append(box_line)

        return np.array(features), box_lines
    
    def load_mmaskes(self, pic_name,raw_img, ):
        path = "/".join(self.box_path.split('/')[:-1]) + "/mask/" + pic_name.replace("/", "_")[:-4] + ".npy"
        maskes = load_roi_mask(path)
        tensor = torch.from_numpy(maskes)
        tensor = tensor.unsqueeze(0)
        target_height, target_width = raw_img.size[1], raw_img.size[0]
        upsampled_tensor = F.interpolate(tensor, size=(target_height, target_width), mode='bilinear',
                                            align_corners=False)
        upsampled_tensor = upsampled_tensor.squeeze(0)
        maskes = upsampled_tensor.numpy()
        return maskes

def get_bboxes_from_file(file_name):
    boxes = []
    file_names = []
    file_boxes = []
    with open(file_name, 'r') as f:
        line = f.readline().strip()
        boxes = []
        while line:
            datas = line.split()
            bbox_name = datas[0].split('>')[0]
            box = list(map(int, datas[1:5]))
            if not file_names:
                file_names.append(bbox_name)
                boxes.append(box)
            else:
                if bbox_name != file_names[-1]:
                    file_names.append(bbox_name)
                    file_boxes.append(boxes)
                    boxes = []
                boxes.append(box)
            line = f.readline().strip()
        file_boxes.append(boxes)

    return file_names, file_boxes

def downsample_mask(mask,h,w,min_num=10, rate=1):
    height, width = mask.shape
    # downsampled_height = int(height // (32*rate))
    # downsampled_width = int(width // (32*rate))
    downsampled_height = h
    downsampled_width = w
    patch_h = height // h
    patch_w = width // w

    downsampled_mask = np.zeros((downsampled_height, downsampled_width), dtype=int)

    for i in range(downsampled_height):
        for j in range(downsampled_width):
            block = mask[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]

            if np.sum(block) > 10:
                downsampled_mask[i, j] = 1

    return downsampled_mask

def load_roi_mask(path):
    mask = np.load(path)
    return mask

def save_bi_np_matrix(matrix, f, is_print=False):
    # with open(file_path, 'wb') as f:
    for i in range(matrix.shape[0]):
        dim = len(matrix[i])
        np.array(dim, dtype=np.uint32).tofile(f)
        matrix[i].astype(np.float32).tofile(f)
    if is_print:
        print(matrix)

def save_str_list(list, file_path, is_print=False):
    if isinstance(file_path, str):
        f = open(file_path, 'w')
    else:
        f = file_path
    for line in list:
        f.write(line)
        if is_print:
            print(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='') 
    parser.add_argument('--dataset_path', type=str, default='./data/test_dataset.txt',help="A text file containing the paths of all the images in the dataset.")
    parser.add_argument('--mask_path', type=str, default='./save/mask/')
    parser.add_argument('--save_path', type=str, default='./save/result_qry/')
    parser.add_argument('--box_path', type=str, default='./data/test_query.txt')
    parser.add_argument('--pooling', type=str, default='gem')  # 'max')
    parser.add_argument('--layer', type=str, default='layer4.1')
    parser.add_argument('--net', type=str, default='resnet50')
    parser.add_argument('--parameter', type=str, default='swav')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--is_qry', type=bool, default=True)
    args = parser.parse_args() 
    print(args)

    feature_extract = Feature_Extraction(args)
    if args.is_qry==False:
        print("ref")
        feature_extract.extract_ref()
    else:
        print("qry")
        feature_extract.extract_qry()

