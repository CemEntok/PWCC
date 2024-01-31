import torch,os,random,time,rawpy,json
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from tqdm import tqdm
from torch import optim
from torch import nn
from model import U_Net
from SVWB_Unet.firasCWCC import CWCC
from SVWB_Unet.utils import *
from metrics import *
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from PIL import Image
from torch.autograd import profiler

class Solver():
    def __init__(self, config, train_loader, valid_loader, test_loader):
        # RAW args
        self.camera = config.camera
        # self.raw = rawpy.imread("/home/cem/LSMI2/"+self.camera+".dng")
        self.white_level = 1023 #self.raw.white_level
        # 
        if self.camera == 'sony':
            self.white_level = self.white_level/4

        # Training config
        self.mode               = config.mode
        self.num_epochs         = config.num_epochs
        self.batch_size         = config.batch_size
        self.lr                 = config.lr
        self.beta1              = config.beta1
        self.beta2              = config.beta2
        self.criterion          = nn.MSELoss(reduction='mean')
        self.TValpha            = config.TValpha
        # self.criterion          = custom_loss
        self.num_epochs_decay   = config.num_epochs_decay
        self.save_epoch         = config.save_epoch
        self.multi_gpu          = config.multi_gpu
        self.gpunumber          = config.gpunumber
        self.jarno              = config.image_pool
        self.model_root         = config.model_root
        self.result_root        = config.result_root

        # Data loader
        self.data_root          = config.data_root
        self.train_loader       = train_loader
        self.valid_loader       = valid_loader
        self.test_loader        = test_loader
        self.input_type         = config.input_type
        self.output_type        = config.output_type
         
        # Models
        self.model_type         = config.model_type
        self.optimizer          = None
        self.img_ch             = config.img_ch
        self.output_ch          = config.output_ch
        self.checkpoint         = config.checkpoint

        # Visualize step
        self.save_result        = config.save_result
        self.vis_step           = config.vis_step
        self.val_step           = config.val_step

        # Misc
        if self.checkpoint:
            self.train_date = self.checkpoint.split('/')[-2] # get base directory from checkpoint
        else:
            self.train_date = time.strftime("%y%m%d_%H%M", time.localtime(time.time()))
            self.train_date = self.train_date.split("_")[0]
            # 
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Uce
    
        # Uce

        # Initialize default path & SummaryWriter
        self.model_path = os.path.join(self.model_root,self.train_date)
        self.result_path = os.path.join(self.result_root,self.train_date)
        if os.path.isdir(self.model_path) == False:
            os.makedirs(self.model_path)
        if os.path.isdir(self.result_path) == False and self.save_result == 'yes':
            os.makedirs(self.result_path)
        if self.mode == "train":
            self.log_path = os.path.join(config.log_root,self.train_date)
            self.writer = SummaryWriter(self.log_path)
            with open(os.path.join(self.model_path,'args.txt'), 'w') as f:
                json.dump(config.__dict__, f, indent=2)
            f.close()

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()

    def build_model(self):
        # build network, configure optimizer
        print("[Model]\tBuilding Network...")
        if self.model_type == "U_Net":
            self.net = U_Net(img_ch=self.img_ch, output_ch=self.output_ch)
        elif self.model_type == "CWCC":
            self.net = CWCC()
        # 
        # Load model from checkpoint
        if self.checkpoint != None:
            # 
            # ckpt_file = 'best.pt' if '/' not in self.checkpoint else os.path.split(self.checkpoint)[1]
            # ckpt = os.path.join(self.model_path,ckpt_file)
            ckpt = self.checkpoint
            print("[Model]\tLoad model from checkpoint :", ckpt)
            # breakpoint()
            self.net.load_state_dict(torch.load(ckpt),strict=False)

        # multi-GPU
        if self.multi_gpu == 1 and torch.cuda.device_count() > 1:
            # self.gpunumber = 1  # Replace with your desired starting value
            # Create a list of numbers from x to 3        
            x = self.gpunumber  # Replace with your desired starting value
            self.GPUList = []
            while x <= 3:
                self.GPUList.append(x)
                x += 1 
            del x
            # 
            self.net = nn.DataParallel(self.net, device_ids=self.GPUList)
            # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # else:
        self.device = torch.device('cuda:'+ str(self.gpunumber) if torch.cuda.is_available() else 'cpu')
            # self.net = nn.DataParallel(self.net)
        # 
        # GPU & optimizer
        self.net = self.net.to(self.device)
        self.optimizer = optim.Adam(list(self.net.parameters()),
                                      self.lr, [self.beta1, self.beta2])
        print("[Model]\tModel Size (MB):", sum(p.numel() for p in self.net.parameters()) * 4 / (1024**2))
        print("[Model]\tBuild Complete.")

    def train(self):
        print("[Train]\tStart training process.")
        torch.set_grad_enabled(True)
        best_net_score = 9876543210.
        best_mae_illum = 9876543210.
        best_psnr = 0.
        best_25worst = 9876543210.
        best_25best  = 9876543210
        early_stopper = EarlyStopper(patience=200, min_delta=0.01)
        # 
        # Training
        # with profiler.profile(use_cuda=True) as prof:
        for epoch in range(self.num_epochs):

                self.net.train()
                # print("Model Size (MB):", sum(p.numel() for p in self.net.parameters()) * 4 / (1024**2))
                trainbatch_len = len(self.train_loader)
                MAE_list = [] # Uce
                train_loss = 0.0
                TVLossTrain = 0.0
                for i, batch in enumerate(self.train_loader):
                    # prepare input
                    if self.input_type == "rgb":
                        inputs = batch["input_rgb"].to(self.device)
                    elif self.input_type == "uvl":
                        inputs = batch["input_uvl"].to(self.device)
                    # prepare GT
                    if self.output_type == "illumination":
                        GTs = batch["gt_illum"].to(self.device)
                    elif self.output_type == "uv":
                        GTs = batch["gt_uv"].to(self.device)
                        # 
                    # prepare mask
                    masks = batch["mask"].to(self.device)
                                  
                    # inference
                    # turning off zero grad because of gradient accumulation technique
                    # self.net.zero_grad() # ADDED BY UCE TO USE PROPER OPTIMIZATION OF THE NETWORK !!!!!!!!!!!!!!!!!!!!!!!!!!
                    # print("Input Size:", inputs.size())
                    # 
                    pred = self.net(inputs) # [20,2,256,256]
                    # 
                    pred_detach = pred.detach()#.cpu() [20,2,256,256]
                    TVLossTrain +=total_variation_loss(pred*masks,self.TValpha).item()
                    pred_loss = self.criterion(pred*masks, GTs*masks) + total_variation_loss(pred*masks,self.TValpha)
                    # 
                    # # tv_loss = total_variation_loss(pred*masks)
                    # 
                    # pred_loss = pred_loss + tv_loss
                    
                    # linear colorline loss
                    if self.output_type == "illumination":
                        illum_map_rb = None
                        # raise NotImplementedError
                    elif self.output_type == "uv":
                        # difference of uv (inputs-pred) equals to illumination RB value
                        illum_map_rb = torch.exp((inputs[:,0:2,:,:]-pred_detach)*masks)

                    # Backprop & optimize network
                    # self.net.zero_grad()# COMMENTED OUT BY UCE TO USE PROPER OPTIMIZATION OF THE NETWORK !!!!!!!!!!!!!!!!!!!!!!!!!!
                    # total_loss = pred_loss 
                    #  

                    self.optimizer.zero_grad()  
                    pred_loss.backward()
                    # Accumulate gradients for a specified number of steps
                    # if (i + 1) % 10 == 0 or i == trainbatch_len - 1:
                        # Update model weights after accumulating gradients
                    self.optimizer.step()
                        
                    train_loss += pred_loss.item() 

                    # calculate pred_rgb & pred_illum & gt_illum
                    input_rgb = batch["input_rgb"].to(self.device)
                    gt_illum = batch["gt_illum"].to(self.device)
                    gt_rgb = batch["gt_rgb"].to(self.device)
                    if self.output_type == "illumination":
                        ones = torch.ones_like(pred_detach[:,:1,:,:])
                        pred_illum = torch.cat([pred_detach[:,:1,:,:],ones,pred_detach[:,1:,:,:]],dim=1)
                        pred_illum[:,1,:,:] = 1.
                        pred_rgb = apply_wb(input_rgb,pred_illum,pred_type='illumination')
                    elif self.output_type == "uv":
                        pred_rgb = apply_wb(input_rgb,pred_detach,pred_type='uv')
                        pred_illum = input_rgb / (pred_rgb + 1e-8)
                        pred_illum[:,1,:,:] = 1.
                    ones = torch.ones_like(gt_illum[:,:1,:,:])
                    gt_illum = torch.cat([gt_illum[:,:1,:,:],ones,gt_illum[:,1:,:,:]],dim=1)

                    # error metrics
                    MAE_illum,_ = get_MAE(pred_illum,gt_illum,tensor_type="illumination",mask=batch["mask"])
                    MAE_rgb,_ = get_MAE(pred_rgb,gt_rgb,tensor_type="rgb",camera=self.camera,mask=batch["mask"])
                    PSNR = get_PSNR(pred_rgb,gt_rgb,white_level=self.white_level)
                    
                    # Uce
                    MAE_list.append(MAE_illum.item()) # uce 
                    # 
                    if math.isnan(train_loss):
                        breakpoint()
                    # 
                    if i % 10 == 9:
                        worst25 = perc25(MAE_list)[0]
                        best25 = perc25(MAE_list)[1]
                        print(f'[Train] Epoch [{epoch+1}/{self.num_epochs}] | ' \
                                f'worst25MAE:{worst25:.5f} | '\
                                f'best25MAE:{best25:.5f} | ')
                                # f'worst25MAE:{worst25.item():.5f} | '\
                                # f'best25MAE:{best25.item():.5f} | ')
                        # # 
                    # Uce
                    
                    # print training log & write on tensorboard & reset vriables
                    print(f'[Train] Epoch [{epoch+1}/{self.num_epochs}] | ' \
                            f'Batch [{i+1}/{trainbatch_len}] | ' \
                            f'train_lossTotal:{train_loss:.5f} | ' \
                            f'pred_loss:{pred_loss.item():.5f} | ' \
                            f'MAE_illum:{MAE_illum.item():.5f} | '\
                            f'MAE_rgb:{MAE_rgb.item():.5f} | '\
                            f'PSNR:{PSNR.item():.5f}')
                    # self.writer.add_scalar('train/total_loss',total_loss,epoch * trainbatch_len + i)
                    # self.writer.add_scalar('train/pred_loss',pred_loss.item(),epoch * trainbatch_len + i)
                    # self.writer.add_scalar('train/MAE_illum',MAE_illum.item(),epoch * trainbatch_len + i)
                    # self.writer.add_scalar('train/MAE_rgb',MAE_rgb.item(),epoch * trainbatch_len + i)
                    # self.writer.add_scalar('train/PSNR',PSNR.item(),epoch * trainbatch_len + i)

                current_memory = torch.cuda.memory_allocated()
                print(f"[Train]\tEpoch {epoch+1}, Batch {i}, GPU Memory Allocated: {current_memory / 1024**3:.2f} GB")
                # 
                # Print or log the average loss for the epoch
                average_loss = train_loss / len(self.train_loader)
                TVLossTrainAvg = TVLossTrain / len(self.train_loader)
                print("-------------------------------------------------------------")
                print(f"[TRAIN] Epoch {epoch + 1}, AVERAGE TRAIN LOSS: {average_loss:.4f}")
                print("-------------------------------------------------------------")

                # lr decay
                if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                    self.lr -= (self.lr / float(self.num_epochs_decay))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr
                    print(f'Decay lr to {self.lr}')
                
                if os.path.isfile(os.path.join(self.model_path,"TrainLosses.txt")) == False:
                    x1 = "w"
                else:
                    x1 = "a"
                with open(os.path.join(self.model_path,"TrainLosses.txt"), x1) as text_file:
                    print(f"AverageTrainLoss:{average_loss}_epoch:{epoch+1}", file=text_file)
                    print(f"AverageTVLossTrain:{TVLossTrainAvg}_epoch:{epoch+1}", file=text_file)
                    print(f"MAE_illum:{MAE_illum}_epoch:{epoch+1}", file=text_file)
                    print(f"MAE_rgb:{MAE_rgb}_epoch:{epoch+1}", file=text_file)

                # Validation
                if epoch % self.val_step != 0:
                    continue
                # print(prof.key_averages().table(sort_by="cuda_time_total"))
                
                torch.set_grad_enabled(False)
                self.net.eval()
                
                valid_loss = 0.0
                valid_pred_loss = 0.0
                valid_MAE_illum = 0
                valid_MAE_rgb = 0
                valid_PSNR = 0
                valid_data_count = 0
                MAE_illumValList = []
                MAE_illumValB25 = [] # uce
                MAE_illumValW25 = [] # uce
                TVLossValid = 0
                # total_loss = 0

                for i, batch in enumerate(self.valid_loader):
                    # batch_size = self.valid_loader.batch_size
                    # prepare input,GT,mask
                    if self.input_type == "rgb":
                        inputs = batch["input_rgb"].to(self.device)
                    elif self.input_type == "uvl":
                        inputs = batch["input_uvl"].to(self.device)
                    if self.output_type == "illumination":
                        GTs = batch["gt_illum"].to(self.device)
                    elif self.output_type == "uv":
                        GTs = batch["gt_uv"].to(self.device)
                    masks = batch["mask"].to(self.device)

                    # inference
                    # print("######### [Validation]\tStart validation process. PATLAMA YOK  BIR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    pred = self.net(inputs)
                    # print("######### [Validation]\tStart validation process. PATLAMA YOK  IKI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    pred_detach = pred.detach()
                    TVLossValid +=(total_variation_loss(pred,self.TValpha)).item()
                    pred_loss = self.criterion(pred, GTs) + total_variation_loss(pred,self.TValpha)
                    # 
                    # linear colorline loss
                    if self.output_type == "illumination":
                        illum_map_rb = None
                        # raise NotImplementedError
                    elif self.output_type == "uv":
                        # difference of uv (inputs-pred) equals to illumination RB value
                        illum_map_rb = torch.exp((inputs[:,0:2,:,:]-pred)*masks)
                    # 
                    valid_loss += pred_loss.item()

                    
                    # calculate pred_rgb & pred_illum & gt_illum
                    input_rgb = batch["input_rgb"].to(self.device)
                    gt_illum = batch["gt_illum"].to(self.device)
                    gt_rgb = batch["gt_rgb"].to(self.device)
                    if self.output_type == "illumination":
                        ones = torch.ones_like(pred_detach[:,:1,:,:])
                        pred_illum = torch.cat([pred_detach[:,:1,:,:],ones,pred_detach[:,1:,:,:]],dim=1)
                        pred_rgb = apply_wb(input_rgb,pred_illum,pred_type='illumination')
                    elif self.output_type == "uv":
                        pred_rgb = apply_wb(input_rgb,pred_detach,pred_type='uv')
                        pred_illum = input_rgb / (pred_rgb + 1e-8)
                    ones = torch.ones_like(gt_illum[:,:1,:,:])
                    gt_illum = torch.cat([gt_illum[:,:1,:,:],ones,gt_illum[:,1:,:,:]],dim=1)
                    pred_rgb = torch.clamp(pred_rgb,0,self.white_level)

                    # error metrics
                    MAE_illum,MAEbatch = get_MAE(pred_illum,gt_illum,tensor_type="illumination",mask=batch["mask"])
                    MAE_rgb,_ = get_MAE(pred_rgb,gt_rgb,tensor_type="rgb",camera=self.camera,mask=batch["mask"])
                    PSNR = get_PSNR(pred_rgb,gt_rgb,white_level=self.white_level)

                    # Uce
                    MAE_illumValList.append(MAE_illum.item()) 
                    worst25,best25 = perc25(MAE_illumValList)
                    # MAE_illumValW25.append(perc25(MAEbatch)[0])
                    MAE_illumValW25.append(0)
                    MAE_illumValB25.append(0)
                    # MAE_illumValB25.append(perc25(MAEbatch)[1])      
                    # Uce

                    valid_pred_loss += (pred_loss * len(inputs))
                    valid_MAE_illum += (MAE_illum.item() * len(inputs))
                    valid_MAE_rgb += (MAE_rgb.item() * len(inputs))
                    valid_PSNR += (PSNR.item() * len(inputs))
                    valid_data_count += len(inputs)

                # 
                valid_loss /= len(self.valid_loader)
                TVLossValidAvg = TVLossValid / len(self.valid_loader)
                valid_pred_loss /= valid_data_count + 1e-16
                valid_MAE_illum /= valid_data_count + 1e-16
                valid_MAE_rgb /= valid_data_count + 1e-16
                valid_PSNR /= valid_data_count + 1e-16
                # 
                if early_stopper.early_stop(valid_pred_loss):             
                    break
                else:
                    print("Early stopper is not activated.")

                if os.path.isfile(os.path.join(self.model_path,"valLosses.txt")) == False:
                    x1 = "w"
                else:
                    x1 = "a"
                with open(os.path.join(self.model_path,"valLosses.txt"), x1) as text_file:
                    if early_stopper.early_stop(valid_loss):
                        print(f"Early stop at epoch: {epoch+1}", file=text_file)
                    print(f"AverageValidPredLoss:{valid_loss}_epoch:{epoch+1}", file=text_file)
                    print(f"AverageValidTVLoss:{TVLossValidAvg}_epoch:{epoch+1}", file=text_file)
                    print(f"Valid_MAE_illum:{valid_MAE_illum}_epoch:{epoch+1}", file=text_file)
                    print(f"valid_MAE_rgb:{valid_MAE_rgb}_epoch:{epoch+1}", file=text_file)

                # print validation log & write on tensorboard every epoch
                print(f'[Valid] Epoch [{epoch+1}/{self.num_epochs}] | ' \
                        f'valid_loss_Avg: {valid_loss:.5f} | ' \
                        f'pred_loss: {valid_pred_loss:.5f} | ' \
                        f'AverageValidTVLoss: {TVLossValidAvg:.5f} | ' \
                        f'MAE_illum: {valid_MAE_illum:.5f} | '\
                        f'MAE_rgb: {valid_MAE_rgb:.5f} | '\
                        f'MAE_illumWorst: {worst25:.5f} | '\
                        f'MAE_illumBest: {best25:.5f} | '\
                        f'MAE_illumWorstv2: {MAE_illumValW25[-1]:.5f} | '\
                        f'MAE_illumBestv2: {MAE_illumValB25[-1]:.5f} | '\
                        f'PSNR: {valid_PSNR:.5f}')
                # self.writer.add_scalar('validation/total_loss',valid_loss,epoch)
                # self.writer.add_scalar('validation/pred_loss',valid_pred_loss,epoch)
                # self.writer.add_scalar('validation/MAE_illum',valid_MAE_illum,epoch)
                # self.writer.add_scalar('validation/MAE_rgb',valid_MAE_rgb,epoch)
                # self.writer.add_scalar('validation/PSNR',valid_PSNR,epoch)
                # self.writer.add_scalar('validation/MAEworst25',worst25,epoch)
                # self.writer.add_scalar('validation/MAEbest25',best25,epoch)

                # Save best U-Net model
                if valid_loss < best_net_score:
                    best_net_score = valid_loss
                    best_net = self.net.module.state_dict()
                    print(f'Best net Score : {best_net_score:.4f}')
                    torch.save(best_net, os.path.join(self.model_path, 'best_loss.pt'))
                    #Uce
                    bestValidPredLoss = "{:.4f}".format(valid_loss)
                    bestLoss_MAE = "{:.2f}".format(valid_MAE_illum)
                    if os.path.isfile(os.path.join(self.model_path,"bests.txt")) == False:
                        x1 = "w"
                    else:
                        x1 = "a"
                    with open(os.path.join(self.model_path,"bests.txt"), x1) as text_file:
                        print(f"best_ValidPredLoss: {bestValidPredLoss} at epoch: {epoch}", file=text_file)
                        print(f"MAE_at_bestValidLoss: {bestLoss_MAE} at epoch: {epoch}", file=text_file)
                    #Uce
                if valid_MAE_illum < best_mae_illum:
                    best_mae_illum = valid_MAE_illum
                    best_net = self.net.module.state_dict()
                    print(f'#####   Best MAE_illum   ##### : {best_mae_illum:.4f}')
                    torch.save(best_net, os.path.join(self.model_path, 'best_mae_illum.pt'))
                    #Uce
                    bestMAE = "{:.4f}".format(best_mae_illum)
                    best_25worst = "{:.4f}".format(worst25)
                    best_25best = "{:.4f}".format(best25)
                    if os.path.isfile(os.path.join(self.model_path,"bests.txt")) == False:
                        x1 = "w"
                    else:
                        x1 = "a"
                    with open(os.path.join(self.model_path,"bests.txt"), x1) as text_file:
                        print(f"bestMAE: {bestMAE} at epoch: {epoch+1}", file=text_file)
                        print(f"best_25worst: {best_25worst} at epoch: {epoch+1}", file=text_file)
                        print(f"best_25best: {best_25best} at epoch: {epoch+1}", file=text_file)
                        print(f"best_25worstv2: {MAE_illumValW25[-1]} at epoch: {epoch+1}", file=text_file)
                        print(f"best_25best: {MAE_illumValB25[-1]} at epoch: {epoch+1}", file=text_file)
                    #Uce
                if valid_PSNR > best_psnr:
                    best_psnr = valid_PSNR
                    best_net = self.net.module.state_dict()
                    print(f'#####   Best PSNR   #####: {best_psnr:.4f}')
                    # torch.save(best_net, os.path.join(self.model_path, 'best_psnr.pt'))
                
                # Save every N epoch
                if self.save_epoch > 0 and epoch % self.save_epoch == self.save_epoch-1:
                    state_dict = self.net.module.state_dict()
                    torch.save(state_dict, os.path.join(self.model_path, str(epoch)+'.pt'))

                # Uce
                # print(f'Best net Score : {best_net_score:.4f}')
                # print(f"bestMAE: {bestMAE:.4f}")
                # print(f"best_25worst: {best_25worst:.4f}")
                # print(f"best_25best: {best_25best:.4f}")
                # print(f'Best PSNR : {best_psnr:.4f}')
                print(f'[VALID]Best net Score : {best_net_score:.4f} | bestMAE: {best_mae_illum:.4f}' \
                    f' | b25(W)MAE: {best_25worst} | b25(B)MAE: {best_25best} | BestPSNR : {best_psnr:.4f}')
                # Uce
                torch.set_grad_enabled(True)
                # Uce
        # print(prof.key_averages().table(sort_by="cuda_time_total"))
    def test(self):
        print("[Test]\tStart testing process.")
        torch.set_grad_enabled(False)
        self.net.eval()

        test_loss = 0.0
        test_loss_avg = 0.0
        test_loss_list = []
        test_pred_loss = []
        test_MAE_illum = []
        test_MAE_rgb = []
        test_MAE_rgb_BF = []
        test_MAE_rgb_BFv2 = []
        test_PSNR = []
        test_data_count = 0
        test_MAE_illum_worst25 = []
        test_MAE_illum_best25 = [] 
        TVLossTest = 0
        test_MAE_illumBF = []
        test_MAE_illumBFv2 = []
        test_MAE_illum_worst25BF = []
        test_MAE_illum_best25BF = [] 
        test_MAE_illum_worst25BFv2 = []
        test_MAE_illum_best25BFv2 = [] 
        dictW25 = {}
        for i, batch in enumerate(self.test_loader):
            # prepare input,GT,mask
            if self.input_type == "rgb":
                inputs = batch["input_rgb"].to(self.device)
            elif self.input_type == "uvl":
                inputs = batch["input_uvl"].to(self.device)
            if self.output_type == "illumination":
                GTs = batch["gt_illum"].to(self.device)
            elif self.output_type == "uv":
                GTs = batch["gt_uv"].to(self.device)
            masks = batch["mask"].to(self.device)

            # inference
            pred = self.net(inputs)
            pred_detach = pred.detach() # will never require gradient
            TVLossTest +=(total_variation_loss(pred,self.TValpha)).item()
            pred_loss = self.criterion(pred, GTs) + total_variation_loss(pred,self.TValpha)
            breakpoint()
            # linear colorline loss
            if self.output_type == "illumination":
                # illum_map_rb = pred = None
                illum_map_rb = pred # dtype=torch.float32
                # raise NotImplementedError
            elif self.output_type == "uv":
                # difference of uv (inputs-pred) equals to illumination RB value
                illum_map_rb = torch.exp((inputs[:,0:2,:,:]-pred)) # torch.float32
            # 
            test_loss += pred_loss.item()
            
            # calculate pred_rgb & pred_illum & gt_illum
            input_rgb = batch["input_rgb"].to(self.device)
            gt_illum = batch["gt_illum"].to(self.device)
            gt_rgb = batch["gt_rgb"].to(self.device)
            ones = torch.ones_like(gt_illum[:,:1,:,:])
            gt_illum = torch.cat([gt_illum[:,:1,:,:],ones,gt_illum[:,1:,:,:]],dim=1)
            if self.output_type == "illumination":
                ones = torch.ones_like(pred_detach[:,:1,:,:])
                pred_illum = torch.cat([pred_detach[:,:1,:,:],ones,pred_detach[:,1:,:,:]],dim=1)
                pred_rgb = apply_wb(input_rgb,pred_illum,pred_type='illumination')
                gt_rgbv2 = apply_wb(batch["input_rgb"].to(self.device),gt_illum,pred_type='illumination')
                gt_rgbv2 = gt_rgbv2.cpu()
            elif self.output_type == "uv":
                pred_rgb = apply_wb(input_rgb,pred_detach,pred_type='uv')
                pred_illum = input_rgb / (pred_rgb + 1e-8)
                                       
            predRGB_uc  =  pred_rgb[0].cpu()
            gtRGB_uc    =  batch["gt_rgb"][0]
            inputRGB_uc =  batch["input_rgb"][0]

            predRGB_uc  =  torch.clamp(predRGB_uc,0,self.white_level)
            gtRGB_uc    =  torch.clamp(gtRGB_uc,0,self.white_level)
            inputRGB_uc =  torch.clamp(inputRGB_uc,0,self.white_level)

            

            pred_rgb = torch.clamp(pred_rgb,0,self.white_level) # 1x3x256x256, torch, iscuda True, self.white_level=1023

            # pred_rgb1 = torch.clamp(pred_rgb,0,self.white_level)
            # pred_rgb2 = boundOut(pred_rgb)  
                   
            # Bilateral Filter applied on illumination map prediction
            bilateralPred = applyBF2(pred_illum,gt_illum)
            pred_rgb_BF = input_rgb / (bilateralPred + 1e-8)
            # BFv2 applied RGB prediction
            bFPredRGB = applyBF2(pred_rgb,gt_rgb)            
            pred_illum_BFv2 = input_rgb / (bFPredRGB + 1e-8)
            pred_rgb_BFv2 = torch.clamp(bFPredRGB,0,self.white_level)
            

            # error metrics
            MAE_illum,MAE_illumVec = get_MAE(pred_illum,gt_illum,tensor_type="illumination",mask=batch["mask"])
            
            # 
            MAE_illumBI,MAE_illumVecBI = get_MAE(bilateralPred,gt_illum,tensor_type="illumination",mask=batch["mask"])
            MAE_illumBIv2,MAE_illumVecBIv2 = get_MAE(pred_illum_BFv2,gt_illum,tensor_type="illumination",mask=batch["mask"])
            MAE_rgb,_ = get_MAE(pred_rgb,gt_rgb,tensor_type="rgb",camera=self.camera,mask=batch["mask"])
            MAE_rgb_BF,_ = get_MAE(pred_rgb_BF,gt_rgb,tensor_type="rgb",camera=self.camera,mask=batch["mask"])
            MAE_rgb_BFv2,_ = get_MAE(pred_rgb_BFv2,gt_rgb,tensor_type="rgb",camera=self.camera,mask=batch["mask"])
            PSNR = get_PSNR(pred_rgb,gt_rgb,white_level=self.white_level)
            # 
            # Uce
            # if i % 10 == 9:
            MAE_illumVecN = MAE_illumVec.cpu().detach().numpy()            
            worst25 = perc25(MAE_illumVecN)[0]
            best25 = perc25(MAE_illumVecN)[1]
            # 
            
            MAE_illumVecN2 = MAE_illumVecBI.cpu().detach().numpy()
            worst25BI = perc25(MAE_illumVecN2)[0]
            best25BI = perc25(MAE_illumVecN2)[1]

            MAE_illumVecBFv2 = MAE_illumVecBIv2.cpu().detach().numpy()
            worst25BIv2 = perc25(MAE_illumVecBFv2)[0]
            best25BIv2 = perc25(MAE_illumVecBFv2)[1]

            # 
            # TVLossTestAvg = TVLossTest / len(self.test_loader)
            print(f'[Test] Batch [{i+1}/{len(self.test_loader)}] | ' \
                        f'test_loss_avg:{test_loss/len(self.test_loader):.5f} | ' \
                        f'TVLossTestAvg:{TVLossTest/(i+1):.5f} | ' \
                        f'MAE_illum:{MAE_illum.item():.5f} | '\
                        f'MAE_illumBI:{MAE_illumBI.item():.5f} | '\
                        f'MAE_rgb:{MAE_rgb.item():.5f} | '\
                        f'MAE_rgb_BF:{MAE_rgb_BF.item():.5f} | '\
                        f'MAE_rgb_BFv2:{MAE_rgb_BFv2.item():.5f} | '\
                        f'worst25MAE:{worst25.item():.5f} | '\
                        f'best25MAE:{best25.item():.5f} | '\
                        f'worst25BI:{worst25BI.item():.5f} | '\
                        f'best25BI:{best25BI.item():.5f} | '\
                        f'PSNR:{PSNR.item():.5f}')
            print(f'MAE_illum:{MAE_illum.item():.5f} | '\
                  f'worst25MAE:{worst25.item():.5f} | '\
                  f'best25MAE:{best25.item():.5f} | ')
            print(f'MAE_illumBI:{MAE_illumBI.item():.5f} | '\
                  f'worst25BI:{worst25BI.item():.5f} | '\
                  f'best25BI:{best25BI.item():.5f} | ')
            test_data_count += len(inputs)
            # 
            
            # print(f'[Test] TVLossAverage: {TVLossTestAvg:.5f}')
            test_loss_list.append(test_loss)
            test_pred_loss.append(pred_loss.item())
            test_MAE_illum.append(MAE_illum.item())
            test_MAE_illum_worst25.append(worst25)

            test_MAE_illum_best25.append(best25)
            test_MAE_rgb.append(MAE_rgb.item())
            test_MAE_rgb_BF.append(MAE_rgb_BF.item())
            test_MAE_rgb_BFv2.append(MAE_rgb_BFv2.item())
            test_PSNR.append(PSNR.item())

            test_MAE_illumBF.append(MAE_illumBI.item())           
            test_MAE_illum_worst25BF.append(worst25BI)
            test_MAE_illum_best25BF.append(best25BI)

            test_MAE_illumBFv2.append(MAE_illumBIv2.item())
            test_MAE_illum_worst25BFv2.append(worst25BIv2)
            test_MAE_illum_best25BFv2.append(best25BIv2)
            # 
            dictW25[batch['img_file'][0]] = [MAE_illum.item(),MAE_illumBI.item(),worst25,worst25BI, best25, best25BI]
            fname_base = batch["place"][0]+'_'+batch["illum_count"][0]
            if os.path.isfile(os.path.join(self.model_path,self.train_date+"_TestError.txt")) == False:
                x1 = "w"
            else:
                x1 = "a"
            with open(os.path.join(self.model_path,self.train_date+"_TestError.txt"), x1) as text_file:
                print(f"[{fname_base}] bestMAE: {MAE_illum.item():.5f}", file=text_file)
                print(f"[{fname_base}] best_25worst: {worst25.item():.5f}", file=text_file)
                print(f"[{fname_base}] best_25best: {best25.item():.5f}", file=text_file)
                print(f"[{fname_base}] bestMAE_BF: {MAE_illumBI.item():.5f}", file=text_file)
                print(f"[{fname_base}] best25BI: {best25BI.item():.5f}", file=text_file)
                print(f"[{fname_base}] worst25BI: {worst25BI.item():.5f}", file=text_file) 

            # 
            if self.save_result == 'yes':
                # desiredPlaces1 = ['Place1034_13.tiff', 
                #                 'Place276_12.tiff',
                #                 'Place940_12.tiff',
                #                 'Place926_12.tiff',
                #                 'Place671_12.tiff',
                #                 'Place588_12.tiff',
                #                 'Place1095_13.tiff',
                #                 'Place1068_13.tiff',
                #                 'Place19_12.tiff',
                #                 'Place275_12.tiff']
                # desiredPlaces =['Place894_12.tiff', # 231228 have different of these compared to base 231222T
                #                 'Place154_12.tiff',
                #                 'Place674_12.tiff',
                #                 'Place1083_13.tiff']
                desiredPlaces = ["Place208_12.tiff","Place588_12.tiff","Place19_12.tiff","Place629_12.tiff","Place957_12.tiff"]
                # # desiredPlaces = desiredPlaces1 + [x for x in desiredPlaces2 if x not in desiredPlaces1]
                if not batch['img_file'][0] in desiredPlaces:
                    print("This NOT included: ", batch['img_file'][0])
                    continue
                else:
                    print("[W25 STAT] This included: ", batch['img_file'][0])
                    # if batch['img_file'][0] =="Place588_12.tiff":
                    #     print("Place588_12")
                    #     


            # print(f'MAE_illum:{MAE_illum.item():.5f} | '\
            #       f'worst25MAE:{worst25.item():.5f} | '\
            #       f'best25MAE:{best25.item():.5f} | ')
            # print(f'MAE_illumBI:{MAE_illumBI.item():.5f} | '\
            #       f'worst25BI:{worst25BI.item():.5f} | '\
            #       f'best25BI:{best25BI.item():.5f} | ')

                # plot illumination map to R,B space
                # if self.jarno != "oneImage" and self.output_type != "illumination":
                # for illum output, illum_map_rb 1x2x256x256 but gt_illum is 1x3x256x256
                # 
                plot_fig,plot_fig_rev = plot_illum(pred_map=illum_map_rb.permute(0,2,3,1).reshape((-1,2)).cpu().detach().numpy(),
                                 gt_map=gt_illum[:,[0,2],:,:].permute(0,2,3,1).reshape((-1,2)).cpu().detach().numpy(),
                                 MAE_illum=MAE_illum,MAE_rgb=MAE_rgb,PSNR=PSNR)
                # UNDER UV OUTPUT
                    # illum_map_rb is 1x2x256x256
                    # abc = illum_map_rb.permute(0,2,3,1).reshape((-1,2))
                    # (Pdb) abc.shape
                    # torch.Size([65536, 2]                                                                                                        )
                    # # 

                if self.jarno != "oneImage":
                    input_srgb, output_srgb, gt_srgb, input_rgb, output_rgb, gt_rgb = visualize(batch['input_rgb'][0],pred_rgb[0],batch['gt_rgb'][0],self.camera,concat=False)
                    _, output_srgbBF, _, _, _, _ = visualize(batch['input_rgb'][0],pred_rgb_BF[0],batch['gt_rgb'][0],self.camera,concat=False)
                    _, output_srgbBFv2, _, _, _, _ = visualize(batch['input_rgb'][0],pred_rgb_BFv2[0],batch['gt_rgb'][0],self.camera,concat=False)
                    
                    if self.output_type == "illumination":
                        input_srgbv2, output_srgbv2, gt_srgbv2, _, _, _ = visualize(inputRGB_uc,predRGB_uc,gtRGB_uc,self.camera,concat=False)
                else:
                    
                    input_srgb, output_srgb, gt_srgb, input_rgb, output_rgb, gt_rgb = visualize(batch['input_rgb'][0],pred_rgb[0],batch['gt_rgb'][0],"jarno",concat=False)

                
                self.train_date = self.checkpoint.split("/")[-2]
                # self.date = self.checkpoint.split("/")[-2]
                
                # if os.path.isfile(os.path.join(self.result_path, fname_base+"_TestError.txt")) == False:
                #     x1 = "w"
                # else:
                #     x1 = "a"
                # with open(os.path.join(self.result_path, fname_base+"_TestError.txt"), x1) as text_file:
                #     print(f"[{fname_base}] bestMAE: {MAE_illum.item():.5f}", file=text_file)
                #     print(f"[{fname_base}] best_25worst: {worst25.item():.5f}", file=text_file)
                #     print(f"[{fname_base}] best_25best: {best25.item():.5f}", file=text_file)
                #     print(f"[{fname_base}] bestMAE_BF: {MAE_illumBI.item():.5f}", file=text_file)
                #     print(f"[{fname_base}] best25BI: {best25BI.item():.5f}", file=text_file)
                #     print(f"[{fname_base}] worst25BI: {worst25BI.item():.5f}", file=text_file)                
                with open(os.path.join(self.result_path, self.train_date+"_" + fname_base + "_TestError.txt"), 'a') as text_file:
                    print(f"[{fname_base}] bestMAE: {MAE_illum.item():.5f}", file=text_file)
                    print(f"[{fname_base}] best_25worst: {worst25.item():.5f}", file=text_file)
                    print(f"[{fname_base}] best_25best: {best25.item():.5f}", file=text_file)
                    print(f"[{fname_base}] bestMAE_BF: {MAE_illumBI.item():.5f}", file=text_file)
                    print(f"[{fname_base}] best25BI: {best25BI.item():.5f}", file=text_file)
                    print(f"[{fname_base}] worst25BI: {worst25BI.item():.5f}", file=text_file)

                if self.jarno=="oneImage":
                    # 
                    Image.fromarray(input_rgb).save(os.path.join(self.result_path,fname_base+'_JARNO_input_rgb.png'))
                    Image.fromarray(gt_rgb).save(os.path.join(self.result_path,fname_base+'_JARNO_input_GTrgb.png'))
                    Image.fromarray(input_srgb).save(os.path.join(self.result_path,fname_base+'_JARNO_input_srgb.png'))
                    Image.fromarray(output_srgb).save(os.path.join(self.result_path,fname_base+'_JARNO_output_srgb.png'))
                    Image.fromarray(gt_srgb).save(os.path.join(self.result_path,fname_base+'_JARNO_gt_srgb.png'))
                    # 
                    normEnder(os.path.join(self.result_path,fname_base+'_JARNO_input_srgb.png'))
                    normEnder(os.path.join(self.result_path,fname_base+'_JARNO_input_srgb.png'))
                    normEnder(os.path.join(self.result_path,fname_base+'_JARNO_input_srgb.png'))

                else:
                    # 
                    Image.fromarray(plot_fig).save(os.path.join(self.result_path,self.train_date+"_"+fname_base+'_illum_map.png'))
                    Image.fromarray(plot_fig_rev).save(os.path.join(self.result_path,self.train_date+"_"+fname_base+'_illum_map_rev.png'))
                    Image.fromarray(input_srgb).save(os.path.join(self.result_path,self.train_date+"_"+fname_base+'_input_srgb.png'))
                    Image.fromarray(output_srgb).save(os.path.join(self.result_path,self.train_date+"_"+fname_base+'_output_srgb.png'))
                    Image.fromarray(output_srgbBF).save(os.path.join(self.result_path,self.train_date+"_"+fname_base+'_output_srgbBF.png'))
                    Image.fromarray(output_srgbBFv2).save(os.path.join(self.result_path,self.train_date+"_"+fname_base+'_output_srgbBFv2.png'))
                    Image.fromarray(gt_srgb).save(os.path.join(self.result_path,self.train_date+"_"+fname_base+'_gt_srgb.png'))
                    
                # 
                pred_illum_scale = pred_illum
                pred_illum_scale[:,1] *= 0.6
                # save_image(fp=os.path.join(self.result_path,fname_base+'_illum.png'),tensor=pred_illum_scale[0].cpu().detach())
                # 
                pred_rgb_normalized = (pred_rgb[0] / self.white_level).cpu().detach()
                gamma_pred_rgb = torch.pow(pred_rgb_normalized,1/1.5)
                # save_image(fp=os.path.join(self.result_path,fname_base+'_raw.png'),tensor=gamma_pred_rgb)
                # 
        np.save(os.path.join(self.model_path,self.checkpoint.split("/")[-2]),dictW25)
        test_loss_avg = test_loss / len(self.test_loader)

        print("average loss :", test_loss_avg)
        print("loss :", np.nanmean(test_loss_list), np.median(test_loss_list), np.max(test_loss_list),perc25(test_loss_list)[0],perc25(test_loss_list)[1])
        print("pred_loss :", np.nanmean(test_pred_loss), np.median(test_pred_loss), np.max(test_pred_loss),perc25(test_pred_loss)[0],perc25(test_pred_loss)[1])
        print("MAE_illum :", np.nanmean(test_MAE_illum), np.median(test_MAE_illum), np.max(test_MAE_illum),perc25(test_MAE_illum)[0],perc25(test_MAE_illum)[1])
        print("MAE_illumBF :", np.nanmean(test_MAE_illumBF), np.median(test_MAE_illumBF), np.max(test_MAE_illumBF),perc25(test_MAE_illumBF)[0],perc25(test_MAE_illumBF)[1])
        print("MAE_illumBFv2 :", np.nanmean(test_MAE_illumBFv2), np.median(test_MAE_illumBFv2), np.max(test_MAE_illumBFv2),perc25(test_MAE_illumBFv2)[0],perc25(test_MAE_illumBFv2)[1])
            
        print("MAE_illumv2 :", np.nanmean(test_MAE_illum), np.median(test_MAE_illum), np.max(test_MAE_illum),np.nanmean(test_MAE_illum_worst25),np.nanmean(test_MAE_illum_best25))
        print("MAE_rgb :", np.nanmean(test_MAE_rgb), np.median(test_MAE_rgb), np.max(test_MAE_rgb),perc25(test_MAE_rgb)[0],perc25(test_MAE_rgb)[1])
        print("MAE_rgb_BF :", np.nanmean(test_MAE_rgb_BF), np.median(test_MAE_rgb_BF), np.max(test_MAE_rgb_BF),perc25(test_MAE_rgb_BF)[0],perc25(test_MAE_rgb_BF)[1])
        print("MAE_rgb_BFv2 :", np.nanmean(test_MAE_rgb_BFv2), np.median(test_MAE_rgb_BFv2), np.max(test_MAE_rgb_BFv2),perc25(test_MAE_rgb_BFv2)[0],perc25(test_MAE_rgb_BFv2)[1])
        print("PSNR :", np.nanmean(test_PSNR), np.median(test_PSNR), np.max(test_PSNR),perc25(test_PSNR)[0],perc25(test_PSNR)[1])
        # 
        if np.isnan(perc25(test_loss_list)[0]) or np.isnan(perc25(test_loss_list)[1]):
            
            print("nan detected in perc25(test_loss)")
        if np.isnan(perc25(test_pred_loss)[0]) or np.isnan(perc25(test_pred_loss)[1]):
            
            print("nan detected in perc25(test_pred_loss)")
        if np.isnan(perc25(test_MAE_illum)[0]) or np.isnan(perc25(test_MAE_illum)[1]):
            
            print("nan detected in perc25(test_MAE_illum)")
        if np.isnan(perc25(test_MAE_rgb)[0]) or np.isnan(perc25(test_MAE_rgb)[1]):
            
            print("nan detected in perc25(test_MAE_rgb)")
        if np.isnan(perc25(test_PSNR)[0]) or np.isnan(perc25(test_PSNR)[1]):
            
            print("nan detected in perc25(test_PSNR)")