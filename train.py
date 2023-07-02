import torch
import scipy.io as sio
import numpy as np
import os
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from utils import *
from modules.network import *
from tqdm import tqdm
from dataset import *
from test import test_model
from torchvision import transforms
import platform
from observation import CDP_operation

parser = ArgumentParser(description='CDP task')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=300, help='epoch number of end training')
parser.add_argument('--hidden_channels', default=64, type=int, help='number of the hidden channels of the network')
parser.add_argument('--stage_numT', type=int, default=5, help='stage number of teacher network')
parser.add_argument('--stage_numS', type=int, default=5, help='stage number of student network')
parser.add_argument('--mask_x', type=int, default=4, help='number of coded masks')
parser.add_argument('--noise_alpha', type=int, default=9, help='strength of noise corruption')

parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--batch_size', type=int, default=8)

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--measurements', type=str, default='CDP_uniform', help='measurement name')

parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--test_name', nargs='+', default='PrDeep12', help='name of test set')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')

parser.add_argument('--optimizer', default='Adam', type=str, help='Currently only support SGD, Adam and Adamw')
parser.add_argument('--scheduler', default='multistep', type=str, help='step | multistep | cosine | cycle')
parser.add_argument('--expe_name', default='CDP_uniformx4', type=str, help='experiment name defined by user')

parser.add_argument('--lr', default=5e-4, type=float, help='initial learning rate')
parser.add_argument('--lr_step', default=50, type=int, help='epochs to decay learning rate by')
parser.add_argument('--gamma', default=0.5, type=float, help='gamma for step scheduler')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD')
parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
parser.add_argument('--beta1', default=0.9, type=float, help='beta1 of Adam')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 of Adam')
parser.add_argument('--eps', default=1e-8, type=float, help='eps of Adam')
parser.add_argument('--weight_decay', default=1e-2, type=float, help='weight decay of optimizer')

parser.add_argument('--eval', action='store_true', help='test or not during training')
parser.set_defaults(eval=False)

args_Parser = Parser(parser)
args = args_Parser.get_arguments()

model_dir = "./%s/CDP_Tstage%d_Sstage%d_lr%.4f_%s" % (args.model_dir, args.stage_numT, args.stage_numS, args.lr, args.expe_name)
log_file_name = "./%s/Log_CDP_Tstage%d_Sstage%d_lr%.4f_%s.txt" % (args.log_dir, args.stage_numT, args.stage_numS, args.lr, args.expe_name)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

args_Parser.write_args(model_dir.split('/')[-1])
args_Parser.print_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

masks_mat_path = './%s/uniform_x4.mat' % (args.matrix_dir)
masks_mat = sio.loadmat(masks_mat_path)['mask']
masks_mat = torch.from_numpy(masks_mat).type(torch.complex64).to(device)
mask_x = args.mask_x
masks_mat = masks_mat[:mask_x, :, :]
assert masks_mat.size(0) == mask_x

T_model = network(args.stage_numT, device, hidden_channels=args.hidden_channels).to(device)
S_model = network(args.stage_numS, device, hidden_channels=args.hidden_channels).to(device)

print('Using Measurements from {}'.format(args.measurements))
measurements_dir = os.path.join(args.data_dir, args.measurements)
dataset = CDPdatasets(measurements_dir)

if (platform.system() =="Windows"):
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, pin_memory=True)
else:
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True)

MSE_Loss = torch.nn.MSELoss()

optimizer_Tmodel = get_Optimizer(args, T_model)
optimizer_Smodel = get_Optimizer(args, S_model)
scheduler_Tmodel = get_Scheduler(args, optimizer_Tmodel)
scheduler_Smodel = get_Scheduler(args, optimizer_Smodel)

r2r_noise_alpha_list = [9,27]
trainS_noise_alpha = args.noise_alpha
testS_noise_alpha = args.noise_alpha
save_interval, eval_interval = 1, 1
spatial_size = [128,128]
transform_aug = transforms.RandomChoice([transforms.RandomResizedCrop(spatial_size[0]), transforms.RandomRotation([-180,180])])

for epoch_i in range(args.start_epoch + 1, args.end_epoch + 1):
    loss_all_list, loss_T_list, loss_S_list = [], [], []
    print('Current epoch: {}'.format(epoch_i))

    for i, data in enumerate(tqdm(data_loader)):

        y = data['y'].to(device)[:, :mask_x, :, :]
        sigma_y = data['sigmas'].to(device)[:, mask_x-1].unsqueeze(-1)
        assert y.size(1) == mask_x and sigma_y.size(1) == 1

        variance = torch.from_numpy(np.array(r2r_noise_alpha_list)[np.random.randint(0, len(r2r_noise_alpha_list), (y.size(0),))]).reshape(y.size(0), 1, 1, 1).to(device)
        r2r_noise = torch.mul(torch.randn_like(y), variance / 255 * y)
        y_masks_plus = torch.pow(y,2) + r2r_noise
        y_masks_plus = torch.sqrt(y_masks_plus * (y_masks_plus > 0))
        y_masks_minus = torch.pow(y,2) - r2r_noise
        y_masks_minus = torch.sqrt(y_masks_minus * (y_masks_minus > 0))
        cond = sigma_y

        x_output, xT_preds = T_model(y_masks_plus, masks_mat, cond, spatial_size)

        loss_T= 0
        for j, each_xout in enumerate(xT_preds):
            y_masks_pred, _ = CDP_operation(each_xout, masks_mat)
            loss_T += MSE_Loss(y_masks_pred, y_masks_minus) / (len(xT_preds) - j)

        optimizer_Tmodel.zero_grad()
        loss_T.backward()
        optimizer_Tmodel.step()

        pil_imgs = []
        for k in range(x_output.size(0)):
            pil_img = transform_aug(torch.clamp(x_output[k,...], max=1, min=0))
            pil_imgs.append(pil_img)
        x2 = torch.stack(pil_imgs, dim=0).to(device).detach()

        y2, cond2 = CDP_operation(x2, masks_mat, trainS_noise_alpha, noise_type='simulatedPoisson')
        x_Soutput, xS_preds = S_model(y2, masks_mat, cond2, spatial_size)

        loss_S = 0
        for k, each_Spred in enumerate(xS_preds):
            loss_S += MSE_Loss(x2, each_Spred) / (len(xS_preds) - k)

        optimizer_Smodel.zero_grad()
        loss_S.backward()
        optimizer_Smodel.step()

        loss_T_list.append(loss_T.item())
        loss_S_list.append(loss_S.item())
        loss_all_list.append((loss_T+loss_S).item())

    scheduler_Tmodel.step()
    scheduler_Smodel.step()

    avg_loss_all = np.array(loss_all_list).mean()
    avg_loss_T = np.array(loss_T_list).mean()
    avg_loss_S = np.array(loss_S_list).mean()
    output_data = "[%02d/%02d] Total Loss: %.4f, Teacher Loss: %.4f, Student Loss: %.4f \n" % (epoch_i, args.end_epoch, avg_loss_all, avg_loss_T, avg_loss_S)
    print(output_data)
    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if epoch_i % save_interval == 0:
        T_states = {'model': T_model.state_dict(), 'optimizer': optimizer_Tmodel.state_dict(), 'epoch': epoch_i}
        torch.save(T_states, "./%s/Tparams_dict_%d.pkl" % (model_dir, epoch_i))
        S_states = {'model': S_model.state_dict(), 'optimizer': optimizer_Smodel.state_dict(), 'epoch': epoch_i}
        torch.save(S_states, "./%s/Sparams_dict_%d.pkl" % (model_dir, epoch_i))

    if args.eval and epoch_i % eval_interval == 0:
        for each_testset in args.test_name:
            mean_PSNR_All, mean_SSIM_All = test_model(args.data_dir, args.result_dir, model_dir.split('/')[-1], each_testset, S_model, epoch_i, testS_noise_alpha, masks_mat, spatial_size, device)
