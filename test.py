import torch
import numpy as np
import os
import glob
import cv2
from scipy import signal
import numpy as np
from observation import *
from utils import *

def psnr(img1,img2):
    PIXEL_MAX = 255.0
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 10 * np.log10(PIXEL_MAX **2 / mse)

def ssim(img1, img2, cut=0, cs_map=False):
    # need [0,255] int

    if cut != 0:
        img1 = img1[cut:-cut, cut:-cut]
        img2 = img2[cut:-cut, cut:-cut]
    if np.max(img1) < 2:
        img1 = img1
        img2 = img2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim.mean()

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def test_model(data_dir, result_dir, model_load_dir, test_name, model, epoch_num, noise_alpha, masks_mat, input_size, device):

    test_dir = os.path.join(data_dir, test_name)
    filepaths = glob.glob(test_dir + '/*.png')
    result_dir = os.path.join(result_dir, test_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    PSNR_Natural = []
    SSIM_Natural = []
    PSNR_Unnatural = []
    SSIM_Unnatural = []
    Natural_Set = ['barbara.png','boat.png','couple.png','peppers.png','cameraman.png','streamandbridge.png']

    ImgNum = len(filepaths)
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)

    img_result_dir = os.path.join(result_dir, model_load_dir, "noise_alpha_%d" % noise_alpha + "_epoch_%d" % (epoch_num))
    if not os.path.exists(img_result_dir):
        os.makedirs(img_result_dir)

    model.eval()
    with torch.no_grad():
        for img_no in range(ImgNum):

            imgName = filepaths[img_no]
            Img = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)
            if Img.shape[0] != input_size[0]:
                Img = cv2.resize(Img, tuple(input_size))

            Iorg = Img.copy()
            Img_output = Iorg / 255.0
            batch_x = torch.from_numpy(Img_output)
            batch_x = batch_x.type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)

            y_masks, cond = CDP_operation(batch_x, masks_mat, noise_alpha, noise_type='simulatedPoisson')
            x_output, _,  = model(y_masks, masks_mat, cond, input_size)

            Prediction_value = x_output.squeeze().cpu().data.numpy()
            X_rec = np.clip(Prediction_value, 0, 1) * 255.0
            rec_PSNR = psnr(X_rec, Iorg.astype(np.float64))
            rec_SSIM = ssim(X_rec, Iorg.astype(np.float64))
            X_rec = np.clip(X_rec, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(img_result_dir,"%s_PSNR_%.2f_SSIM_%.4f.png" % (os.path.basename(imgName), rec_PSNR, rec_SSIM)), X_rec)

            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM
            if test_name.lower() == 'prdeep_12_128':
                if os.path.basename(imgName) in Natural_Set:
                    PSNR_Natural.append(rec_PSNR)
                    SSIM_Natural.append(rec_SSIM)
                else:
                    PSNR_Unnatural.append(rec_PSNR)
                    SSIM_Unnatural.append(rec_SSIM)
    model.train()
    mean_PSNR_All = np.mean(PSNR_All)
    mean_SSIM_All = np.mean(SSIM_All)

    if test_name.lower() == 'prdeep_12_128':
        PSNR_Unnatural_avg = np.array(PSNR_Unnatural).mean()
        SSIM_Unnatural_avg = np.array(SSIM_Unnatural).mean()
        PSNR_Natural_avg = np.array(PSNR_Natural).mean()
        SSIM_Natural_avg = np.array(SSIM_Natural).mean()

        output_data = "Epoch: %d, Testset: %s, alpha: %d, Avg PSNR/SSIM: %.4f/%.4f, Natural: %.4f/%.4f, Unnatural: %.4f/%.4f.\n" % (epoch_num, test_name, noise_alpha, mean_PSNR_All, mean_SSIM_All, PSNR_Natural_avg, SSIM_Natural_avg, PSNR_Unnatural_avg, SSIM_Unnatural_avg)
    else:
        output_data = "Epoch: %d, Testset: %s, alpha: %d, Avg PSNR/SSIM: %.4f/%.4f.\n" % (epoch_num, test_name, noise_alpha, mean_PSNR_All, mean_SSIM_All)

    message = output_data

    print(message)
    output_file_name = os.path.join(result_dir, model_load_dir, 'PSNR_SSIM_Results.txt')
    if not os.path.exists(os.path.join(result_dir, model_load_dir)):
        os.makedirs(os.path.join(result_dir, model_load_dir))
    output_file = open(output_file_name, 'a')
    output_file.write(message)
    output_file.close()

    return mean_PSNR_All, mean_SSIM_All