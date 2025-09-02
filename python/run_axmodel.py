import os
import cv2
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm

import common
import imgproc
import axengine as axe

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="edsr_baseline_x2_1.axmodel", help="axmodel model path")
parser.add_argument('--scale', nargs='+', type=int, default=[2], help='super resolution scale')
parser.add_argument("--dir_demo", type=str, default='../video/test_1920x1080.mp4', help="demo image directory")
parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return np.round(np.clip(img * pixel_range, 0, 255)) / pixel_range

def from_numpy(x):
    return x if isinstance(x, np.ndarray) else np.array(x)

class VideoTester():
    def __init__(self, scale, my_model, dir_demo, rgb_range=255, cuda=True, arch='EDSR'):
        self.scale = scale
        self.rgb_range = rgb_range
        self.session = axe.InferenceSession(my_model, 'AxEngineExecutionProvider')
        self.output_names = [x.name for x in self.session.get_outputs()]
        self.input_name = self.session.get_inputs()[0].name
        self.dir_demo = dir_demo  
        self.filename, _ = os.path.splitext(os.path.basename(dir_demo))
        self.arch = arch

    def test(self):
        torch.set_grad_enabled(False)
        if not os.path.exists('experiment'):
            os.makedirs('experiment')
        for idx_scale, scale in enumerate(self.scale):
            vidcap = cv2.VideoCapture(self.dir_demo)
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

            vidwri = cv2.VideoWriter(
                os.path.join('experiment', ('{}_x{}.avi'.format(self.filename, scale))),
                cv2.VideoWriter_fourcc(*'XVID'),
                vidcap.get(cv2.CAP_PROP_FPS),
                (
                    int(scale * vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(scale * vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
            )

            total_times = 0
            tqdm_test = tqdm(range(total_frames), ncols=80)
            
            if self.arch == 'EDSR':
                for _ in tqdm_test:
                    success, lr = vidcap.read()
                    if not success: break
                    start_time  = time.time()
                    lr_y_image, = common.set_channel(lr, n_channels=3)
                    lr_y_image, = common.np_prepare(lr_y_image, rgb_range=self.rgb_range)
                    
                    sr = self.session.run(self.output_names, {self.input_name: lr_y_image})
                    end_time = time.time()
                    total_times += end_time - start_time
                    
                    if isinstance(sr, (list, tuple)):
                        sr = from_numpy(sr[0]) if len(sr) == 1 else [from_numpy(x) for x in sr]
                    else:
                        sr = from_numpy(sr)

                    sr = quantize(sr, self.rgb_range).squeeze(0)
                    normalized = sr * 255 / self.rgb_range
                    ndarr = normalized.transpose(1, 2, 0).astype(np.uint8)
                    vidwri.write(ndarr)
                    
            elif self.arch == 'ESPCN':
                for _ in tqdm_test:
                    success, lr = vidcap.read()
                    if not success: break
                    start_time  = time.time()
                    
                    lr_y_image, lr_cb_image, lr_cr_image = imgproc.preprocess_one_frame(lr)
                    bic_cb_image = cv2.resize(lr_cb_image,
                              (int(lr_cb_image.shape[1] * scale),
                                int(lr_cb_image.shape[0] * scale)),
                              interpolation=cv2.INTER_CUBIC)
                    bic_cr_image = cv2.resize(lr_cr_image,
                              (int(lr_cr_image.shape[1] * scale),
                                int(lr_cr_image.shape[0] * scale)),
                              interpolation=cv2.INTER_CUBIC)
                              
                    sr = self.session.run(self.output_names, {self.input_name: lr_y_image})
                    end_time = time.time()
                    total_times += end_time - start_time
                    
                    if isinstance(sr, (list, tuple)):
                        sr = from_numpy(sr[0]) if len(sr) == 1 else [from_numpy(x) for x in sr]
                    else:
                        sr = from_numpy(sr)

                    ndarr = imgproc.array_to_image(sr)
                    sr_y_image = ndarr.astype(np.float32) / 255.0
                    sr_ycbcr_image = cv2.merge([sr_y_image[:, :, 0], bic_cb_image, bic_cr_image])
                    sr_image = imgproc.ycbcr_to_bgr(sr_ycbcr_image)
                    sr_image = np.clip(sr_image* 255.0, 0 , 255).astype(np.uint8)
                    vidwri.write(sr_image)
                    
            print('Total time: {:.3f} seconds for {} frames'.format(total_times, total_frames))
            print('Average time: {:.3f} seconds for each frame'.format(total_times / total_frames)) 

            vidcap.release()
            vidwri.release()

        torch.set_grad_enabled(True)

def main():
    args = parser.parse_args()
    t = VideoTester(args.scale, args.model, args.dir_demo, arch='EDSR')
    t.test()

if __name__ == '__main__':
    main()
