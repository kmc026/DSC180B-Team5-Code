import cv2
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from skimage import filters
from skimage.data import camera
from skimage.util import compare_images
import skimage.io
from skimage import measure

def write_ssim(image, gt_image, result_dir):
    bs = cv2.imread(image, 1)
    bs = cv2.cvtColor(bs, cv2.COLOR_RGBA2BGR)
    bs = cv2.cvtColor(bs, cv2.COLOR_RGBA2GRAY)
    bs = cv2.GaussianBlur(bs, (5,5), 0)

    bs_gt = cv2.imread(gt_image, 1)
    bs_gt = cv2.cvtColor(bs_gt, cv2.COLOR_RGBA2BGR)
    bs_gt = cv2.cvtColor(bs_gt, cv2.COLOR_RGBA2GRAY)
    bs_gt = cv2.resize(bs_gt, (bs.shape[1], bs.shape[0]))
    bs_gt = bs_gt / bs_gt.max()

    # LoG
    bs_lap = cv2.Laplacian(bs, cv2.CV_64F)
    bs_lap = bs_lap/bs_lap.max()
    # Sobel X
    bs_sob = cv2.Sobel(bs, cv2.CV_64F, 1, 0, ksize=5)
    bs_sob = bs_sob / bs_sob.max()
    # Canny
    bs_can = cv2.Canny(bs, 100, 200)
    bs_can = bs_can / bs_can.max()
    # Prewitt
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    bs_prew = cv2.filter2D(bs, -1, kernelx)
    bs_prew = bs_prew / bs_prew.max()
    # Roberts
    bs_rob = filters.roberts(bs)
    bs_rob = bs_rob / bs_rob.max()

    bs_ssim_lap = skimage.metrics.structural_similarity(bs_gt, bs_lap)
    bs_ssim_sob = skimage.metrics.structural_similarity(bs_gt, bs_sob)
    bs_ssim_can = skimage.metrics.structural_similarity(bs_gt, bs_can)
    bs_ssim_prew = skimage.metrics.structural_similarity(bs_gt, bs_prew)
    bs_ssim_rob = skimage.metrics.structural_similarity(bs_gt, bs_rob)

    f = open(result_dir, "a")
    f.write('SSIM of Image and Ground Truth' + '\n')
    f.write('Laplacian: ' + str(bs_ssim_lap) + '\n')
    f.write('SobelX: ' + str(bs_ssim_sob) + '\n')
    f.write('Canny: ' + str(bs_ssim_can) + '\n')
    f.write('Prewitt: ' + str(bs_ssim_prew) + '\n')
    f.write('Roberts: ' + str(bs_ssim_rob) + '\n')
    f.write('\n')