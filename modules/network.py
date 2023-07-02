import torch
import torch.nn as nn
from .unet import UNet
from .conv_lstm import ConvLSTMCell

class condition_network(nn.Module):
    def __init__(self, inchannels, stage_num, hidden_channels=64):
        super(condition_network, self).__init__()
        self.fc1 = nn.Linear(inchannels, hidden_channels, bias=True)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels, bias=True)
        self.fc3 = nn.Linear(hidden_channels, stage_num * 2, bias=True)
        self.BN = nn.BatchNorm1d(1)
        self.act12 = nn.ReLU(inplace=True)
        self.act3 = nn.Softplus()

    def forward(self, x):
        x = self.BN(x)
        x = self.act12(self.fc1(x))
        x = self.act12(self.fc2(x))
        x = self.act3(self.fc3(x))
        num=x.shape[1]
        num=int(num/2)
        return x[:,0:num], x[:,num:]

class BasicBlock(torch.nn.Module):
    def __init__(self, hidden_channels=64, padding_mode='zeros', kernel_size = 3, norm_layer=nn.InstanceNorm2d):
        super(BasicBlock, self).__init__()
        pad = (kernel_size-1) //2
        if norm_layer is nn.LayerNorm:
            norm_layer_object = nn.LayerNorm([hidden_channels,128,128])
        else:
            norm_layer_object = norm_layer(hidden_channels)

        self.cathconv = nn.Sequential(*[nn.Conv2d(2*hidden_channels, hidden_channels, kernel_size, 1, pad, bias=True, padding_mode=padding_mode), norm_layer_object, nn.LeakyReLU(negative_slope=0.2, inplace=True)])
        self.head_conv = nn.Sequential(*[nn.Conv2d(2, hidden_channels, kernel_size, 1, pad, bias=True, padding_mode=padding_mode), norm_layer_object, nn.LeakyReLU(negative_slope=0.2, inplace=True)])
        self.mid_block = UNet(hidden_channels, hidden_channels, kernel_size=kernel_size, norm_layer=norm_layer if norm_layer is not nn.LayerNorm else nn.InstanceNorm2d, pad=padding_mode, concat_x=False, need_sigmoid=False, feature_scale=4, upsample_mode='deconv')
        self.tail_conv = nn.Sequential(*[nn.Conv2d(hidden_channels, 1, kernel_size, 1, pad, bias=True, padding_mode=padding_mode), nn.LeakyReLU(negative_slope=0.2, inplace=True)])

    def forward(self, x, ys, masks, lambda_step, sigma, h):
        DM_x =torch.fft.fft2(torch.mul(masks.unsqueeze(0).repeat(x.size(0), 1, 1, 1), x.repeat(1, masks.size(0), 1, 1)),norm='ortho')
        abs_Ax = torch.abs(DM_x)
        delta_y = ys - abs_Ax
        D_gradx = (torch.conj(masks.unsqueeze(0).repeat(x.size(0), 1, 1, 1)) * torch.fft.ifft2((DM_x / (abs_Ax+1e-16) * (-delta_y)), norm='ortho')).real
        x = x - lambda_step.view(-1, 1, 1, 1) * torch.mean(D_gradx, dim=1, keepdim=True)
        
        x_input = x
        sigma = torch.ones_like(x) * sigma.view(x.size(0),1,1,1)
        x = self.head_conv(torch.cat((x, sigma), dim=1))
        x = self.cathconv(torch.cat((x, h), dim=1))
        x = self.mid_block(x)
        x_mid = x
        x = self.tail_conv(x)
        x_tail = x
        x = x_input + x
        fms = torch.cat((x_input, x_mid, x_tail), dim=1)

        return x, fms

class network(torch.nn.Module):
    def __init__(self, stage_num, device, hidden_channels=64):
        super(network, self).__init__()
        onelayer = []
        self.stage_num = stage_num
        self.device = device
        for i in range(stage_num):
            onelayer.append(BasicBlock(hidden_channels=hidden_channels))

        self.fcs = nn.ModuleList(onelayer)
        self.condition = condition_network(inchannels=1, stage_num=stage_num, hidden_channels=hidden_channels)
        self.convlstm_cells = nn.ModuleList([ConvLSTMCell(input_dim=hidden_channels+2, hidden_dim=hidden_channels, kernel_size=(3,3), bias=True) for _ in range(stage_num-1)])


    def forward(self, ys, masks, cond, input_size):
        x = torch.ones(ys.size(0), 1, input_size[0], input_size[1]).to(self.device)
        x_pred_list = []
        lambda_step, sigma = self.condition(cond)

        h, c = self.convlstm_cells[0].init_hidden(ys.size(0), input_size)
        for i in range(self.stage_num):
            x, fms = self.fcs[i](x, ys, masks, lambda_step[:,i], sigma[:,i], h)
            if i < self.stage_num - 1:
                h, c = self.convlstm_cells[i](fms, (h,c))
            x_pred_list.append(x)

        return x, x_pred_list
