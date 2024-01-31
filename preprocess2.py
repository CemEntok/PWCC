"""
This code
1) splits train/validation/test places,
2) generates MCC chart area mask (mask.png),
3) subtracts default black level for each camera,
3) then crops & resizes the LSMI dataset
"""

import cv2,os,shutil,json
import rawpy
import numpy as np
from tqdm import tqdm
from utils import *
import warnings
warnings.filterwarnings("error")
SQUARE_CROP = True      # Trim left/right sides of the image so that it is square. (for train/val only, Always True for test)
SIZE = 512              # Size of train/val image. If None, keep the original resolution.
TEST_SIZE = 256         # Size of test image. If None, keep the original resolution.
CAMERA = "/mnt/ssd-storage/cem/dataset/galaxy"       # LSMI subset camera
if SIZE != None:
    DST_ROOT = CAMERA + "_" + str(SIZE) + "_noisyGT"
else:
    DST_ROOT = CAMERA + "_fullres"
ZERO_MASK = -1          # Zero mask value for black pixels
mixmapNoisyDict = {}

RAW = CAMERA+".dng"
TEMPLETE = rawpy.imread(RAW)
if CAMERA == "sony":
    BLACK_LEVEL = 128
else:
    BLACK_LEVEL = min(TEMPLETE.black_level_per_channel)
SATURATION = TEMPLETE.white_level


with open(os.path.join(CAMERA,"meta.json"), 'r') as meta_json:
    meta_data = json.load(meta_json)
with open(os.path.join(CAMERA,"split.json"), 'r') as split_json:
    split_data = json.load(split_json)

for key, places in split_data.items(): # key for example, 'two_illum_train'
    split = key.split("_")[-1]
    print("Processing "+key)

    for place in tqdm(places): # place for example, 'Place100'
        # place = "Place1092" #Uce
        # breakpoint()
        print(place)
        if int(place.split("Place")[-1])>1000:
            print(f"Place {place} and afterwards is not used for training with noisy GORUND TRUTH")
            break
        files = [f for f in os.listdir(os.path.join(CAMERA, place)) if f.endswith("tiff")]
        dst_path = os.path.join(DST_ROOT,split)
        if os.path.isdir(dst_path) == False:
            os.makedirs(dst_path)

        for file in files: # file for example, 'Place100_12.tiff'
            # file = "Place1092_123.tiff"
            # if "two_illum" in key and "_12.tiff" not in file:
            #     continue
            if "three_illum" in key and "_123.tiff" not in file:
                continue
            # Uce
            if "three_illum" in key: # exclude three illumination images
                continue
            # Uce

            fname = os.path.splitext(file)[0]
            illum_count = fname.split("_")[1]

            # open tiff image & subtract black level
            img = cv2.cvtColor(cv2.imread(os.path.join(CAMERA,place,file), cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB).astype('float32')
            img = np.clip(img - BLACK_LEVEL, 0, SATURATION)
            
            # make pixel-level illumination map
            if len(illum_count) == 1:
                mixmap = np.ones_like(img[:,:,0:1],dtype=float)
            else:
                mixmap = np.load(os.path.join(CAMERA,place,fname+".npy"))
                if mixmap.shape[2] == 3:
                    breakpoint()
                mixmapNew = insertNoise(mixmap[:,:,0])
                keyMixmap = f'{key}_{place}'
                mixmapNewMeanOverPixel = mean_matrix = np.mean(mixmapNew, axis=(0, 1), keepdims=True)
                mixmapNoisyDict[keyMixmap] = mixmapNewMeanOverPixel
                mixmap = mixmapNew
                del mixmapNew
                # breakpoint()
                # mixmap = insertNoise(alpha)
            if len(np.where(mixmap<0)[0]) != 0:
                abc = mixmap[np.where(mixmap<0)]
                for k in abc:
                    if k != -1:
                        breakpoint()
            
            illum_chroma = [[0,0,0],[0,0,0],[0,0,0]]
            for i in illum_count:
                illum_chroma[int(i)-1] = meta_data[place]["Light"+i]
            illum_map = mix_chroma(mixmap,illum_chroma,illum_count)
            x_zero,y_zero = np.where(mixmap[:,:,0]==ZERO_MASK)
            illum_map[x_zero,y_zero,:] = [1.,1.,1.]

            # # white balance original image
            # img_wb = img / illum_map
            # img_wb = img / (illum_map + EPS)
            # if place == "Place712":
            #     breakpoint()
            # Uce
            try:
                # white balance original image
                img_wb = img / illum_map           
                # abc = 0
                # for k in range(3):
                #     abc += len(np.where(illum_map == 0)[k])
                # if abc:
                #     breakpoint()

            except RuntimeWarning:
                # breakpoint()
                print("RuntimeWarning: invalid value encountered in true_divide")
                # Workaorund #3
                x_zero,y_zero,z_zero = np.where(illum_map==0)
                for k in range(len(x_zero)):
                    illum_map[x_zero[k],y_zero[k],z_zero[k]] = 1.0
                #Workaround #2
                # def minNonzero(input):
                #     row,col = np.where( input==np.min(input[np.nonzero(input)]))
                #     value = input[row[0],col[0]]  
                #     print(f"row = {row[0]+1} and col = {col[0]+1}")
                #     print(f"min non zero value = {value}")
                #     return row,col,value
                # _,_,val = minNonzero(mixmap[:,:,0])
                try:
                    img_wb = img / illum_map 
                except: 
                    print("[AGAIN] RuntimeWarning: invalid value encountered in true_divide")
                    breakpoint()
            # Uce
            # breakpoint()
            # apply MCC mask to original image, GT image (training set)
            if split == "train":
                mask = np.ones_like(img[:,:,0:1], dtype='float32')
                mcc1 = (np.float32(meta_data[place]["MCCCoord"]["mcc1"]) / 2).astype(int)
                mcc2 = (np.float32(meta_data[place]["MCCCoord"]["mcc2"]) / 2).astype(int)
                mcc3 = (np.float32(meta_data[place]["MCCCoord"]["mcc3"]) / 2).astype(int)
                mcc_list = [mcc1.tolist(),mcc2.tolist(),mcc3.tolist()]
                for mcc in mcc_list:
                    contour = np.array([[mcc[0]],[mcc[1]],[mcc[2]],[mcc[3]]]).astype(int)
                    cv2.drawContours(mask, [contour], 0, (0), -1)
                img = img * mask
                img_wb = img_wb * mask

            # Crop original image, GT image, mixmap
            if SQUARE_CROP or split == "test":
                height, width, _ = img.shape # 1500,2000
                w_start = int(width/2) - int(height/2)
                w_end = w_start + height
                img = img[:,w_start:w_end,:]
                img_wb = img_wb[:,w_start:w_end,:]
                mixmap = mixmap[:,w_start:w_end,:]
                # prevent negative mask value interpolation if ZERO_MASK is negative value
                mixmap = np.where(mixmap==ZERO_MASK,0,mixmap)
            
            # resize & save
            if split == 'test':
                resize_len = TEST_SIZE
            else:
                resize_len = SIZE

            if resize_len != None: # resize_len = 512
                img = cv2.resize(img, dsize=(resize_len,resize_len), interpolation=cv2.INTER_LINEAR).astype('uint16') # (512, 512, 3)
                img_wb = cv2.resize(img_wb, dsize=(resize_len,resize_len), interpolation=cv2.INTER_LINEAR).astype('uint16') # (512, 512, 3)
                mixmap = cv2.resize(mixmap, dsize=(resize_len,resize_len), interpolation=cv2.INTER_LINEAR) # mixmap = (512, 512, 2)
            else:
                img = img.astype('uint16')
                img_wb = img_wb.astype('uint16')
                
            # save image, GT image, MCC mask, mixmap
            cv2.imwrite(os.path.join(dst_path,file), cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(dst_path,fname+"_gt.tiff"), cv2.cvtColor(img_wb,cv2.COLOR_RGB2BGR))
            if split == "train":
                if SQUARE_CROP:
                    mask = mask[:,w_start:w_end,:]
                if resize_len != None:
                    mask = cv2.resize(mask, dsize=(resize_len,resize_len), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(dst_path,place+"_mask.png"), mask)
            if len(illum_count) != 1:
                np.save(os.path.join(dst_path,fname), mixmap)

        # delete original MCC coordinates in meta json
        meta_data[place].pop("MCCCoord")
new_dict = {key: value.astype(np.float32) for key, value in mixmapNoisyDict.items()}
np.save(os.path.join(CAMERA,'mixmapNoisyDict.npz'), mixmapNoisyDict)
np.save(os.path.join(CAMERA,'mixmapNoisyDict32.npz'), new_dict)
# loaded_dict = np.load('results_dict.npy', allow_pickle=True).item()

# shutil.copy(os.path.join(CAMERA,"meta.json"),DST_ROOT)
with open(os.path.join(DST_ROOT,"meta.json"), 'w') as out_file:
    json.dump(meta_data, out_file, indent=4)
shutil.copy(os.path.join(CAMERA,"split.json"),DST_ROOT)
