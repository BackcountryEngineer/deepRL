import numpy as np
import cv2

from collections import deque

# preprocessing
def preprocess_frame(frame):
    gray = np.dot(frame, [0.2989, 0.5870, 0.1140])
    cropped_frame = gray[8:-12, 4:-12]
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = cv2.resize(normalized_frame, (84, 110))
    return preprocessed_frame

# Stack multiple frames into one state
def stack_frames(state, is_new_episode=False, stack=None, stack_size=4):
    frame = preprocess_frame(state)
    
    if is_new_episode:
        stack = deque([frame for i in range(stack_size)], maxlen=stack_size)
        stacked_state = np.stack(stack, axis=2)
    else:
        stack.append(frame)
        stacked_state = np.stack(stack, axis=2)

    return stacked_state, stack
