from PIL import Image
import torchvision.transforms as T
import numpy as np
from collections import deque

##进行图像帧压缩；原图像属于210 * 160 * 3；
transfor = T.Compose([
    T.ToPILImage(),
    T.Grayscale(),    ##将channel从之前的三维RGB图像转化为灰度图像；
    T.Resize((84, 84)),
    T.ToTensor()
])
##进行转化的框架；
def get_frames(frame):
    return transfor(frame)

def stack_frames(stacked_frame, new_frame, is_new_episode):
    if is_new_episode:
        stacked_frame = deque([np.zeros((84, 84), dtype=np.float32)]*4, maxlen=4)
        for _ in range(4):
            stacked_frame.append(new_frame)
    else:
        stacked_frame.append(new_frame)
    stacked_state = np.stack(stacked_frame, axis=0)
    return stacked_state, stacked_frame