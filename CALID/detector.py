import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as pth_transforms
import torch
import os
import sys
import torch.nn.functional as F
from vit_feature import get_dino_output
from scipy import ndimage
import random
from sklearn.datasets import make_blobs
from tqdm import tqdm
import argparse


class CLAID():
    def __init__(self,args):
        self.gpu = args.gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
        self.device = torch.device("cuda:" + self.gpu)

        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        self.patch_size = 8
        self.model.eval()
        self.model.to(self.device)
        
    def extract_dataset(self, dataset_path,init_way,save_path):
        assert dataset_path != None
        assert save_path != None

        path = save_path  + "/mask"
        if not os.path.exists(path):
            os.makedirs(path)
        
        frd = open(dataset_path, 'r')
        fsave = open(save_path + "/box_ref.txt", "a")
        mask_save_path = os.path.join(save_path, 'mask')
        lines = frd.readlines()

        with torch.no_grad():
            for idx, line in enumerate(tqdm(lines)):
                img_path = line.strip()
                self.img, self.rate = self.get_input_var(img_path, show_img=False)
                img = self.img.to(self.device)
                w, h = img.shape[1] - img.shape[1] % self.patch_size, img.shape[2] - img.shape[2] % self.patch_size
                img = img[:, :w, :h].unsqueeze(0)
                output, attens,q, k, v = self.model.get_last_output_and_selfattention(img)
                output = output[0, 1:, :]
                self.updata_total_param(output)

                bp_plts, bp_objects = self.k_sum_main(init_way)

                boxes = []
                maskes = []
                
                objects_list = [item for sublist in bp_objects for item in sublist]
                plts_list = [item for sublist in bp_plts for item in sublist]

                for object,mask in zip(objects_list, plts_list):
                    if len(object) != 0:
                        box = self.get_objects_coordinate(object)
                        boxes.append(box)
                        maskes.append(mask)

                self.save_prm_and_box_list( maskes, boxes, fsave, mask_save_path, img_path)
    
    def get_objects_coordinate(self, obj):
        assert obj.shape[0] == 2
        y_c, x_c = obj
        ymin = int(min(y_c) * self.patch_size * self.rate)
        ymax = int(max(y_c) * (self.patch_size+1) * self.rate)
        xmin = int(min(x_c) * self.patch_size * self.rate)
        xmax = int(max(x_c) * (self.patch_size+1) * self.rate)
        return xmin, ymin, xmax, ymax

    def k_sum_one(self, img_path,init_way,save_path):
        with torch.no_grad():
            self.img, self.rate = self.get_input_var(img_path, show_img=False)
            img = self.img.to(self.device)
            w, h = img.shape[1] - img.shape[1] % self.patch_size, img.shape[2] - img.shape[2] % self.patch_size
            img = img[:, :w, :h].unsqueeze(0)
            output, attens,q, k, v = self.model.get_last_output_and_selfattention(img)
            output = output[0, 1:, :]
            self.updata_total_param(output)

            total_plt, total_object = self.k_sum_main(init_way)
        
        visualize = True
        if visualize:
            directory = save_path +"/" +img_path.split("/")[-1].replace(".jpg","")
            directory_all = directory + "/all/"
            directory_layer = directory + "/layer/"
            if not os.path.exists(directory_all):
                os.makedirs(directory_all)
            
            if not os.path.exists(directory_layer):
                os.makedirs(directory_layer)

            for ite in range(len(total_plt)):
                sub = total_plt[ite]
                seg_map = np.zeros(sub[0].shape)
                print("the num of ",ite," is: ",len(sub))

                for ite_sub in range(len(sub)):
                    seg_map += 2*(ite_sub+20)*sub[ite_sub]

                plt.imshow(seg_map)
                plt.colorbar()
                plt.savefig(directory_layer+str(ite)+".jpg")
                plt.clf() 
            
            for ite in range(len(total_plt)):
                sub = total_plt[ite]
                seg_map = np.zeros(sub[0].shape)
                print("the num of ",ite," is: ",len(sub))

                for ite_sub in range(len(sub)):
                    seg_map = sub[ite_sub]

                    plt.imshow(seg_map)
                    plt.savefig(directory_all+str(ite)+"_"+str(ite_sub)+".jpg")
                    plt.clf() 
        return 0
    
    def k_sum_main(self,init_way):
        total_plt = []
        total_object = []
        n, d = self.dims
        p_list_list =[np.array([i for i in range(n*d)])]

        while p_list_list!=[]:      
            new_p_list_list =[]
            sub_plt = []
            sub_object = []

            dis_func = self.ksum_Is
            
            for p_list_o in p_list_list:
                L, set1, set2 = dis_func(p_list_o.tolist(),init_way)
                if L.any()==False:
                    continue

                fore_objects, fore_plt = self.get_objects(set1, self.dims)
                back_objects, back_plt = self.get_objects(set2, self.dims)

                #using purity
                temp_list = []
                for a in range(len(fore_plt)):
                    plt_a = fore_plt[a]
                    p_list = np.ravel_multi_index(np.where(plt_a==1), self.dims)
                    connect = (np.sum(self.correspend[np.ix_(p_list,p_list)]))/(len(p_list)*(len(p_list)-1))
                    count = np.sum(self.l1_labels[p_list])
                    purity = count / len(p_list)
                    if connect<0.97:
                        temp_list.append(p_list)
                    if purity>0.1:
                        sub_plt.append(plt_a)
                        sub_object.append(fore_objects[a])
                    
                for a in range(len(back_plt)):
                    plt_a = back_plt[a]
                    p_list = np.ravel_multi_index(np.where(plt_a==1), self.dims)
                    connect = (np.sum(self.correspend[np.ix_(p_list,p_list)]))/(len(p_list)*(len(p_list)-1))
                    count = np.sum(self.l1_labels[p_list])
                    purity = count / len(p_list)
                    if connect<0.97:
                        temp_list.append(p_list)
                    if purity>0.1:
                        sub_plt.append(plt_a)
                        sub_object.append(back_objects[a])
                new_p_list_list += temp_list
            
            p_list_list = new_p_list_list
            total_plt.append(sub_plt)
            total_object.append(sub_object)
        
        if [] in total_plt:
            total_plt.remove([])
        if [] in total_object:
            total_object.remove([])
        return total_plt, total_object
    
    def save_prm_and_box_list(self, prm_list, box_list, fsave, mask_save_path, img_path, save_info=True):
        name = img_path.split('/')[-1].split('.')[0]

        prm_np = np.array(prm_list).astype(np.uint8)
        if save_info:
            file_name = mask_save_path + "/" + name + ".npy"
            # print(file_name)
            np.save(file_name, prm_np)
        else:
            print(prm_np)

        for i in range(len(box_list)):
            info = img_path + ">" + str(i).rjust(4, '0') + " " + " ".join(str(num) for num in box_list[i]) + "\n"
            if save_info:
                fsave.write(info)
            else:
                print(info)

    def updata_total_param(self, output):
        self.output = F.normalize(output, p=2).detach().cpu().numpy()
        self.l1norm = torch.sum(torch.abs(output),dim=1)
        taothre = torch.sort(self.l1norm)[0][len(self.l1norm)*7//10]
        self.l1_labels = torch.where(self.l1norm>=taothre,1,0).detach().cpu().numpy()       
        self.h_featmap = self.img.shape[-2] // self.patch_size
        self.w_featmap = self.img.shape[-1] // self.patch_size
        self.dims = [self.h_featmap, self.w_featmap]
        self.correspend = (self.output @ self.output.transpose(1, 0))
        self.di_diag = np.diagonal(self.correspend)
        self.no_binary_graph = False
        self.tau = 0.2
        self.eps = 1e-5
        if self.no_binary_graph:
            self.correspend[self.correspend < self.tau] = self.eps
        else:
            self.correspend = self.correspend > self.tau
            self.correspend = np.where(self.correspend.astype(float) == 0, self.eps, self.correspend)
        self.d_i = np.sum(self.correspend, axis=1)
        self.constraint = -1
    
    def get_input_var(self, img_path, show_img=False):
        if img_path is not None:
            with open(img_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                if show_img:
                    plt.imshow(img)
                    plt.axis('off')
                    plt.show()
        else:
            print(f"Provided image path {img_path} is non valid.")
            sys.exit(1)

        w, h = img.size

        if min(w, h) > 355:
            transform = pth_transforms.Compose([
                pth_transforms.Resize(360, max_size=800),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            # rate = max(w, h) / 640
        else:
            transform = pth_transforms.Compose([
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        img = transform(img)
        rate = h / img.shape[-2]

        return img, rate
    
    def calculate_connect(self, p_list1, p_list2):
        A = self.correspend[np.ix_(p_list1, p_list2)]
        d_i = np.sum(A)
        return d_i
    
    def ksum_Is(self, p_list, init_way = "aim",k=2, iters=500):
        _, dim = self.output.shape
        n = len(p_list)
        feats = self.output
        # # init cluster
        D = np.zeros((k, dim))  # synthetic vector 
        D_ftf = np.zeros((k))
        N = np.zeros(k) # number of members in each cluster
        L = np.zeros(n, dtype=int) # to which cluster each feature belongs
        FtF = self.di_diag

        if init_way=="random":
            for j in range(n):
                point = p_list[j]
                r_num = random.randint(0, k - 1)
                L[j] = r_num
                D[r_num] += feats[point]
                N[r_num] += 1
                D_ftf[r_num] += FtF[point]
        else:
            A = self.correspend[np.ix_(p_list, p_list)]
            tau = 0.2
            eps = 1e-5
            d_i = np.sum(A, axis=1)
            sorted_indices = np.argsort(d_i)
            min_index =sorted_indices[0]
            max_index =sorted_indices[-1]
            L = np.where((A[max_index]-A[min_index])>0,1, 0)
            for j in range(n):
                point = p_list[j]
                r_num = L[j]
                D[r_num] += feats[point]
                N[r_num] += 1
                D_ftf[r_num] += FtF[point]

        step = 1000
        iter=0
        while step!=0 and iter<=iters:
            iter += 1
            keys = [i for i in range(n)]
            random.shuffle(keys)
            step = 0
            for kid in range(len(keys)):
                idx = keys[kid] 
                label = L[idx]
                feat = feats[p_list[idx]]

                now_dw = N[label]*FtF[p_list[idx]] -2*feat@np.transpose(D[label]) + D_ftf[label]
                D_temp = np.zeros(k)
                D_temp[label] = FtF[p_list[idx]]
                D_ftf_i = D_ftf - D_temp
                
                dv = N*FtF[p_list[idx]]-2*feat@np.transpose(D) + D_ftf_i
                Imm = now_dw-dv
                current_Im = Imm.max()
                target_label = np.argmax(Imm)

                if current_Im > 0 and label!=target_label:
                    step+=1
                    D[label], N[label], D[target_label], N[target_label] = self.update_para(D[label], N[label],
                                                                                            D[target_label],
                                                                                            N[target_label], feat)
                    D_ftf[label] -= FtF[p_list[idx]]
                    D_ftf[target_label] += FtF[p_list[idx]]
                    L[idx] = target_label

        fore = np.array(p_list)[np.where(L==0)[0]]
        back = np.array(p_list)[np.where(L==1)[0]]
        return L, fore, back
    
    def get_objects(self, index, dims, min_patch_num=15):
        idx = np.unravel_index(index, dims)
        bipartition = np.zeros(dims)
        bipartition[idx[0], idx[1]] = 1

        s = [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]
        objects, num_objects = ndimage.label(bipartition, structure=s)

        object_list = []
        plt_list = []
        for idx in range(num_objects):
            cc = np.array(np.where(objects == idx + 1))
            if cc.shape[1] < min_patch_num:
                continue
            object_list.append(cc)
            mask = np.zeros(dims)
            mask[cc[0], cc[1]] = 1
            plt_list.append(mask)
        return object_list, plt_list
    
    def update_para(self, D_old, n_old, D_new, n_new, feat):
        # print(D_old.shape, n_old, D_new.shape, D_new)
        D_old -= feat
        n_old -= 1
        D_new += feat
        n_new += 1
        # print(D_old.shape, n_old, D_new.shape, D_new)
        return D_old, n_old, D_new, n_new

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='') 
    parser.add_argument('--mode', type=str, default='dataset',help="extract one or a dataset (choices=[one,dataset])", choices=["one","dataset"])
    parser.add_argument('--dataset_path', type=str, default='./data/test_dataset.txt',help="A text file containing the paths of all the images in the dataset.")
    parser.add_argument('--init', type=str, default='aim', help="the way to initialize ksum (choices=[random,aim])",choices=["random","aim"])
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--image_path', type=str, default='./data/test_image.jpg')
    parser.add_argument('--save_path', type=str, default='./save/')
    args = parser.parse_args() 
    print(args)

    claid = CLAID(args)
    if args.mode=="one":
        img_path = args.image_path
        init_way = args.init
        save_path = args.save_path
        claid.k_sum_one(img_path,init_way,save_path)
    else:
        dataset_path = args.dataset_path
        init_way = args.init
        save_path = args.save_path
        claid.extract_dataset(dataset_path,init_way,save_path)
   

