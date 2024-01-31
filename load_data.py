import os,cv2,json,colorsys,time,sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from SVWB_Unet.utils import *
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

class LSMI(data.Dataset):
    def __init__(self,root,split,image_pool,
                 input_type='uvl',output_type=None,uncalculable=-1,
                 mask_uncalculable=None,mask_highlight=None,mask_black=None,
                 illum_augmentation=None,transform=None,noisyGT=False,batch_size=32,num_epochs=2000):
        self.root = root                        # dataset root
        self.split = split                      # train / val / test
        self.image_pool = image_pool            # 1 / 2 / 3
        self.input_type = input_type            # uvl / rgb
        self.output_type = output_type          # None / illumination / uv
        self.uncalculable = uncalculable        # Masked value for uncalculable mixture
        self.mask_uncalculable = mask_uncalculable  # None or Masking value for uncalculable mixture
        self.mask_highlight = mask_highlight    # None or Saturation value
        self.mask_black = mask_black            # masking value for G=0 pixels
        self.random_color = illum_augmentation
        self.transform = transform
        # self.model_root = "/home/cem/LSMI2/SVWB_Unet/models"
        self.image_list = []
        self.noisyGT = noisyGT
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        # self.device = torch.device('cuda:'+ str(self.gpunumber) if torch.cuda.is_available() else 'cpu')      
        current_file_path = os.path.abspath(sys.argv[0])
        self.model_root = os.path.join(current_file_path.split(os.path.basename(current_file_path))[0],"models")
        # 
        if self.image_pool=="default":
            self.image_pool = [1,2,3]
            self.image_list = sorted([f for f in os.listdir(os.path.join(self.root,self.split))
                                    if f.endswith(".tiff")
                                    and len(os.path.splitext(f)[0].split("_")[-1]) in self.image_pool
                                    and 'gt' not in f])

        elif self.image_pool=="multiv2":      
            # if self.image_pool == "multi":
            self.image_list = []
            a = 0
            listPlace = os.listdir(os.path.join(self.root,self.split))
            listPlaceP = [k for k in listPlace if not k.endswith("_gt.tiff") and not k.endswith(".npy")]
            # 
            for k in listPlaceP:

                if k.endswith("123.tiff"):
                    self.image_list.append(k)
                elif k.endswith("12.tiff"):
                    if k.split("_")[0] + "_123.tiff" not in listPlaceP:
                        self.image_list.append(k)
                        # print(k)
            # 

        elif self.image_pool=="multi":      
            # if self.image_pool == "multi":
            self.image_list = []
            a = 0
            listPlace = os.listdir(os.path.join(self.root,self.split))
            listPlaceP = [k for k in listPlace if not k.endswith("_gt.tiff") and not k.endswith(".npy")]

            for k in listPlaceP:

                if k.endswith("13.tiff"):
                    self.image_list.append(k)
                elif k.endswith("12.tiff"):
                    if k.split("_")[0] + "_123.tiff" not in listPlaceP:
                        self.image_list.append(k)
                        # print(k)
            # 
        elif self.image_pool == "single":
            # elif self.image_pool == "single":
            self.image_list = []
            listPlace = os.listdir(os.path.join(self.root,self.split))
            listPlaceP = [k for k in listPlace if not k.endswith("_gt.tiff") and not k.endswith(".npy")]
            for k in listPlaceP:
                if k.endswith("1.tiff"):
                    if k.split("_")[0] + "_123.tiff" not in listPlaceP:
                        self.image_list.append(k)
                        # print(k)
        elif self.image_pool == "oneImage":
            self.image_list = []
            self.root = "/mnt/ssd-storage/cem/dataset/"
            self.split = "VisualTest"
            img =  next(k for k in os.listdir(os.path.join(self.root,self.split)) if k.endswith(".tiff"))
            self.image_list.append(img)

        # if self.image_pool == "default":
        meta_file = os.path.join(self.root,'meta.json')
        if self.image_pool == "oneImage":
            meta_file = os.path.join(self.root,self.split,'testJ.json')
        with open(meta_file, 'r') as meta_json:
            self.meta_data = json.load(meta_json)
        # 
        print("[Data]\t"+str(self.__len__())+" "+split+" images are loaded from "+root)
        self.mixmapNoisyDict = {}
        self.mixmapDict = {}
        # self.idxCount = 0
        # self.epoch = 0
        self.train_date = time.strftime("%y%m%d_%H%M", time.localtime(time.time()))
        # self.train_date= self.train_date.split("_")[1]
        # self.saveN = 0
        self.idxCount = 0
        if split == "train":
            self.trainDataSize = self.__len__()
            self.iterperEpoch  = self.trainDataSize // self.batch_size +1
            # self.iterperEpoch = self.trainDataSize//self.batch_size + 1 
            # self.total_iterations = self.batch_size * self.iterperEpoch
    def __getitem__(self, idx):
        """
        Returns
        metadata        : meta information
        input_***       : input image (uvl or rgb)
        gt_***          : GT (None or illumination or chromaticity)
        mask            : mask for undetermined illuminations (black pixels) or saturated pixels
        """

        # parse fname
        fname = os.path.splitext(self.image_list[idx])[0]
        img_file = fname+".tiff"
        mixmap_file = fname+".npy"
        place, illum_count = fname.split('_')
        #print(place,illum_count)

        # 1. prepare meta information
        ret_dict = {}
        ret_dict["illum_chroma"] = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
        for illum_no in illum_count:
            illum_chroma = self.meta_data[place]["Light"+illum_no]
            ret_dict["illum_chroma"][int(illum_no)-1] = illum_chroma  # illum_chroma = [R1/G1, 1, B1/G1; R2/G2, 1, B2/G2]
        # 
        ret_dict["img_file"] = img_file
        ret_dict["place"] = place
        ret_dict["illum_count"] = illum_count

        # 2. prepare input & output GT
        # load mixture map & 3 channel RGB tiff image
        input_path = os.path.join(self.root,self.split,img_file)
        input_bgr = cv2.imread(input_path, cv2.IMREAD_UNCHANGED).astype('float32')
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
        # 
        if len(illum_count) >1:
            mixmap = np.load(os.path.join(self.root,self.split,mixmap_file)).astype('float32')
            # 
            mixmapNew = mixmap.copy()
        elif len(illum_count) == 1:
            mixmap = np.ones_like(input_rgb[:,:,0:1])
            # no adding noise to alpha when image is single illumination
            mixmapNew = mixmap.copy()
        # ADD NOISE - UCE
        if self.noisyGT and self.split == "train":
            if len(illum_count) == 2:
                mixmapNew = insertNoise(mixmap[:,:,0],weight=self.noisyGT)
            keyMixmap = f'{fname}'
            mixmapNewMeanOverPixel = np.mean(mixmapNew, axis=(0, 1), keepdims=True)
            mixmapMeanOverPixel = np.mean(mixmap, axis=(0, 1), keepdims=True)
            self.mixmapNoisyDict[keyMixmap] = mixmapNewMeanOverPixel
            self.mixmapDict[keyMixmap] = mixmapMeanOverPixel
            mixmap = mixmapNew
            # print(f"[NOISE]\tSAVING THE NOISY DATA")
            # if os.path.isdir(os.path.join(self.model_root,"NoiseData")) == False:
            #     os.makedirs(os.path.join(self.model_root,"NoiseData"))
            # np.savez(os.path.join(os.path.join(self.model_root,"NoiseData"),"ALphasNoises2.npz"), dict1=self.mixmapDict, dict2=self.mixmapNoisyDict)


            # 
        
        # mixmap contains -1 for ZERO_MASK, which means uncalculable pixels with LSMI's G channel approximation.
        # So we must replace negative values to 0 if we use pixel level augmentation.
        uncalculable_masked_mixmap = np.where(mixmap==self.uncalculable,0.0,mixmap) # mixmap = [alpha, 1-alpha]
        # breakpoint()
        # Uce
        # uncalculable_masked_mixmap = np.where(mixmap<0,0.0,mixmap)
        # Uce
        # random data augmentation
        if self.random_color and self.split=='train':
            augment_chroma = self.random_color(illum_count)
            ret_dict["illum_chroma"] *= augment_chroma
            tint_map = mix_chroma(uncalculable_masked_mixmap,augment_chroma,illum_count)
            breakpoint()
            if countNeg(tint_map)>0: # Uce
                print("[ERROR - WARNING]\tNegative value in tint map!")               
                # tint_map[tint_map < 0] = 0
                # print("[WARNING]\tNegative values in tint map are limited to 0")
                # 
                if os.path.isfile(os.path.join(self.model_root,"tintMapErr.txt")) == False:
                    # # 
                    x1 = "w"
                else:
                    x1 = "a"
                with open(os.path.join(self.model_root,"tintMapErr.txt"), x1) as text_file:
                    print(f"Tint map error in Image: {img_file}\n", file=text_file)
            # 
            input_rgb = input_rgb * tint_map    # apply augmentation to input image

        # prepare input tensor
        ret_dict["input_rgb"] = input_rgb
        ret_dict["input_uvl"] = rgb2uvl(input_rgb)
        
        # prepare output tensor

        illum_map = mix_chroma(uncalculable_masked_mixmap,ret_dict["illum_chroma"],illum_count)
        ret_dict["gt_illum"] = np.delete(illum_map, 1, axis=2)
        output_bgr = cv2.imread(os.path.join(self.root,self.split,fname+"_gt.tiff"), cv2.IMREAD_UNCHANGED).astype('float32')
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        ret_dict["gt_rgb"] = output_rgb
        output_uvl = rgb2uvl(output_rgb)
        ret_dict["gt_uv"] = np.delete(output_uvl, 2, axis=2) # delete the luminanance channel
        # if self.noisyGT and self.split == "train":
            
        # 
        # illum_map = mix_chroma(uncalculable_masked_mixmap,ret_dict["illum_chroma"],illum_count) # l_ab = alpha*l_a+(1-alpha)*l_b, (256,256,3), [R/G, 1, B/G]
        # ret_dict["gt_illum"] = np.delete(illum_map, 1, axis=2)
        # if self.output_type == 'illumination':
        #     # pass  
        #     illum_map = torch.from_numpy(illum_map.reshape(illum_map.shape[-1],illum_map.shape[0], illum_map.shape[1]))
        #     illum_map = illum_map.unsqueeze(0)
        #     # illum_map = np.expand_dims(illum_map, axis=0)  # Shape: (1, 3)         
        #     ones = torch.ones_like(illum_map[:,:1,:,:])
        #     gt_illumCalc = torch.cat([illum_map[:,:1,:,:],ones,illum_map[:,1:,:,:]],dim=1)
        #     output_rgb = apply_wb(torch.from_numpy(input_rgb.reshape(input_rgb.shape[-1],input_rgb.shape[0], input_rgb.shape[1])).unsqueeze(0),gt_illumCalc,pred_type='illumination')
        #     abc = output_rgb.squeeze()
        #     numpy_array = abc.numpy().copy()
        #     output_rgb = numpy_array.reshape(numpy_array.shape[1],numpy_array.shape[2], numpy_array.shape[0])
        #     # ret_dict["gt_rgb"] = output_rgb
        # elif self.output_type == 'uv':
        #     output_bgr = cv2.imread(os.path.join(self.root,self.split,fname+"_gt.tiff"), cv2.IMREAD_UNCHANGED).astype('float32')
        #     output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        # ret_dict["gt_rgb"] = output_rgb # 256,256,3, numpy ndarray
        # output_uvl = rgb2uvl(output_rgb) # 
        # ret_dict["gt_uv"] = np.delete(output_uvl, 2, axis=2)

        # 3. prepare mask
        if self.split == 'train':
            mask = cv2.imread(os.path.join(self.root,self.split,place+'_mask.png'), cv2.IMREAD_GRAYSCALE)
            mask = mask[:,:,None].astype('float32')
        else:
            mask = np.ones_like(input_rgb[:,:,0:1], dtype='float32')
        if self.mask_uncalculable != None:
            mask[mixmap[:,:,0]==self.uncalculable] = self.mask_uncalculable
        if self.mask_highlight != None:
            raise NotImplementedError("Implement highlight masking!")
        if self.mask_black != None:
            mask[input_rgb[:,:,1:2]==0] = self.mask_black
        ret_dict["mask"] = mask
        
        # 4. apply transform
        if self.transform != None:
            ret_dict = self.transform(ret_dict)
                                
        return ret_dict

    def __len__(self):
        return len(self.image_list)

class PairedRandomCrop():
    def __init__(self,size=(256,256),scale=(0.3,1.0),ratio=(1.,1.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
    def __call__(self,ret_dict):
        i,j,h,w = RandomResizedCrop.get_params(img=ret_dict['input_rgb'],scale=self.scale,ratio=self.ratio)
        ret_dict['input_rgb'] = TF.resized_crop(ret_dict['input_rgb'],i,j,h,w,self.size)
        ret_dict['input_uvl'] = TF.resized_crop(ret_dict['input_uvl'],i,j,h,w,self.size)
        ret_dict['gt_illum'] = TF.resized_crop(ret_dict['gt_illum'],i,j,h,w,self.size)
        ret_dict['gt_rgb'] = TF.resized_crop(ret_dict['gt_rgb'],i,j,h,w,self.size)
        ret_dict['gt_uv'] = TF.resized_crop(ret_dict['gt_uv'],i,j,h,w,self.size)
        ret_dict['mask'] = TF.resized_crop(ret_dict['mask'],i,j,h,w,self.size)
        
        return ret_dict

class Resize():
    def __init__(self,size=(256,256)):
        self.size = size
    def __call__(self,ret_dict):
        ret_dict['input_rgb'] = TF.resize(ret_dict['input_rgb'],self.size)
        ret_dict['input_uvl'] = TF.resize(ret_dict['input_uvl'],self.size)
        ret_dict['gt_illum'] = TF.resize(ret_dict['gt_illum'],self.size)
        ret_dict['gt_rgb'] = TF.resize(ret_dict['gt_rgb'],self.size)
        ret_dict['gt_uv'] = TF.resize(ret_dict['gt_uv'],self.size)
        ret_dict['mask'] = TF.resize(ret_dict['mask'],self.size)

        return ret_dict

class ToTensor():
    def __call__(self, ret_dict):
        ret_dict['input_rgb'] = torch.from_numpy(ret_dict['input_rgb'].transpose((2,0,1)))
        ret_dict['input_uvl'] = torch.from_numpy(ret_dict['input_uvl'].transpose((2,0,1)))
        ret_dict['gt_illum'] = torch.from_numpy(ret_dict['gt_illum'].transpose((2,0,1)))
        ret_dict['gt_rgb'] = torch.from_numpy(ret_dict['gt_rgb'].transpose((2,0,1)))
        ret_dict['gt_uv'] = torch.from_numpy(ret_dict['gt_uv'].transpose((2,0,1)))
        ret_dict['mask'] = torch.from_numpy(ret_dict['mask'].transpose((2,0,1)))
        
        return ret_dict

class RandomColor():
    def __init__(self,sat_min,sat_max,val_min,val_max,hue_threshold):
        self.sat_min = sat_min
        self.sat_max = sat_max
        self.val_min = val_min
        self.val_max = val_max
        self.hue_threshold = hue_threshold

    def hsv2rgb(self,h,s,v):
        return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))
    
    def threshold_test(self,hue_list,hue):
        if len(hue_list) == 0:
            return True
        for h in hue_list:
            if abs(h - hue) < self.hue_threshold:
                return False
        return True

    def __call__(self, illum_count):
        hue_list = []
        ret_chroma = [[0,0,0],[0,0,0],[0,0,0]]
        for i in illum_count:
            while(True):
                hue = np.random.uniform(0,1)
                saturation = np.random.uniform(self.sat_min,self.sat_max)
                value = np.random.uniform(self.val_min,self.val_max)
                chroma_rgb = np.array(self.hsv2rgb(hue,saturation,value), dtype='float32')
                chroma_rgb /= chroma_rgb[1]

                if self.threshold_test(hue_list,hue):
                    hue_list.append(hue)
                    ret_chroma[int(i)-1] = chroma_rgb
                    break

        return np.array(ret_chroma)


def get_loader(config, split):
    random_color = None
    if config.illum_augmentation == 'yes' and split=='train':
        random_color = RandomColor(config.sat_min,config.sat_max,
                                   config.val_min,config.val_max,
                                   config.hue_threshold)

    random_crop = None
    if config.random_crop=='yes' and split=='train':
        # train mode & random crop
        tsfm = transforms.Compose([ToTensor(),
                                   PairedRandomCrop(size=(config.image_size,config.image_size),scale=(0.3,1.0),ratio=(1.,1.))])
    elif config.image_size != None:
        # validation & test mode or train mode without random crop / square resizing
        tsfm = transforms.Compose([ToTensor(),
                                   Resize(size=(config.image_size,config.image_size))])
    else :
        # validation & test mode or train mode without random crop / original image size
        tsfm = transforms.Compose([ToTensor()])

    dataset = LSMI(root=config.data_root,
                   split=split,
                   image_pool=config.image_pool,
                   input_type=config.input_type,
                   output_type=config.output_type,
                   uncalculable=config.uncalculable,
                   mask_uncalculable=config.mask_uncalculable,
                   mask_black=config.mask_black,
                   mask_highlight=config.mask_highlight,
                   illum_augmentation=random_color,
                   transform=tsfm,
                   noisyGT=config.noisyGT,
                   batch_size=config.batch_size,
                   num_epochs=config.num_epochs)
    
    if split == 'test':
        dataloader = data.DataLoader(dataset,batch_size=1,shuffle=False,
                                     num_workers=config.num_workers)
    else:
        dataloader = data.DataLoader(dataset,batch_size=config.batch_size,
                                     shuffle=True,num_workers=config.num_workers)

    return dataloader

if __name__ == "__main__":
    random_color = RandomColor(0.2,0.8,1,1,0.2)
    random_crop = PairedRandomCrop(size=(512,512),scale=(0.3,1.0),ratio=(1.,1.))
    resize = Resize((256,256))
    tsfm = transforms.Compose([ToTensor(),
                               random_crop])

    data_set = LSMI(root='galaxy_512',
                      split='test',image_pool="custom",
                      input_type='uvl',output_type='uv',
                      uncalculable=-1,illum_augmentation=random_color,
                      transform=tsfm)

    data_loader = data.DataLoader(data_set, batch_size=1, shuffle=False)

    for batch in data_loader:
        print(batch["img_file"])
        print(batch["illum_chroma"])
        print(batch["input_uvl"].shape)
        print(batch["gt_uv"].shape)
        print(batch["mask"].shape)

        # illum_map_rb = (batch["gt_illum"]).permute(0,2,3,1).reshape((-1,2))
        # illum_img = plot_illum(illum_map_rb)
        # cv2.imwrite('visualize/'+batch["place"][0]+"_"+batch["illum_count"][0]+'.png',illum_img)
        input()