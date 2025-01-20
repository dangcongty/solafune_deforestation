import cv2
import os
import numpy as np
from scipy.ndimage import rotate



class Augmentation:
    def __init__(self):
        pass

    def flip_lr(self, img: np.ndarray, mask: np.ndarray, p: float):
        '''
        input:
            img: CxHxW
            mask: HxW
        '''
        if np.random.random() > p:
            return img[:, :, ::-1], mask[:, ::-1]
        else:
            return img, mask

    def flip_ud(self, img: np.ndarray, mask: np.ndarray, p: float):
        '''
        input:
            img: CxHxW
            mask: HxW
        '''
        if np.random.random() > p:
            return img[:, ::-1, :], mask[::-1, :]
        else:
            return img, mask

    def rotate(self, img: np.ndarray, mask: np.ndarray, angle: int, p: float):
        def apply(array, angle):
            C, H, W = array.shape
            cx, cy = W // 2, H // 2  # Center of the image
            theta = np.radians(angle) # Convert angle to radians
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ]) # Rotation matrix (2D)
            y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij') # Create a grid of (x, y) coordinates
            # Shift coordinates to center, apply rotation, then shift back
            coords = np.stack([x - cx, y - cy], axis=-1)  # Shape: (H, W, 2)
            rotated_coords = np.dot(coords, R.T).astype(np.float32)  # Apply rotation
            rotated_coords[..., 0] += cx  # Shift back
            rotated_coords[..., 1] += cy
            valid_mask = (
                (rotated_coords[..., 0] >= 0) & (rotated_coords[..., 0] < W) &
                (rotated_coords[..., 1] >= 0) & (rotated_coords[..., 1] < H)
            ) # Identify out-of-bounds coordinates
            
            rotated_coords = np.clip(rotated_coords, 0, [W - 1, H - 1]).astype(int)  # Clip coordinates to valid range
            # Generate rotated image by mapping original pixel values
            rotated_array = np.zeros_like(array)
            for c in range(C):
                rotated_array[c] = array[c, rotated_coords[..., 1], rotated_coords[..., 0]]
            rotated_array[:, ~valid_mask] = 0
            return rotated_array
        
        if np.random.random() > p:
            angle = np.random.randint(-angle, angle)
            rotated_image = apply(img, angle)
            rotated_mask = apply(mask[None, :, :], angle)[0]

            return rotated_image, rotated_mask
        else:
            return img, mask

    def translate(self, img: np.ndarray, mask: np.ndarray, shift_ratio: int, p: float):
        if np.random.random() > p:
            percent_shift = np.random.randint(0, shift_ratio) # phần trăm shift

            shift_direct = np.random.choice(['lr', 'td', 'both']) # shift top-down or left-right?
            shifted_img = np.zeros_like(img) # C x H x W
            shifted_mask = np.zeros_like(mask) # H x W
            h, w = shifted_mask.shape

            if shift_direct == 'lr':
                shift = int(percent_shift*w/100)
                if np.random.random() > 0.5: # shift left
                    shifted_img[:, :, :w-shift] = img[:, :, shift:]
                    shifted_mask[:, :w-shift] = mask[:, shift:]
                else:
                    shifted_img[:, :, shift:] = img[:, :, :w-shift]
                    shifted_mask[:, shift:] = mask[:, :w-shift]

            elif shift_direct == 'ud':
                shift = int(percent_shift*h/100)
                if np.random.random() > 0.5: # shift down
                    shifted_img[:, :h-shift, :] = img[:, shift:, :]
                    shifted_mask[:h-shift, :] = mask[shift:, :]
                else:
                    shifted_img[:, shift:, :] = img[:, :h-shift, : ]
                    shifted_mask[shift:, :] = mask[:h-shift, : ]

            else:
                shift_w = int(percent_shift*h/100)
                shift_h = int(percent_shift*w/100)
                if np.random.random() > 0.5: # shift left
                    shifted_img[:, :, :w-shift_w] = img[:, :, shift_w:]
                    shifted_mask[:, :w-shift_w] = mask[:, shift_w:]
                else:
                    shifted_img[:, :, shift_w:] = img[:, :, :w-shift_w]
                    shifted_mask[:, shift_w:] = mask[:, :w-shift_w]

                if np.random.random() > 0.5: # shift down
                    shifted_img[:, :h-shift_h, :] = shifted_img[:, shift_h:, :]
                    shifted_mask[:h-shift_h, :] = shifted_mask[shift_h:, :]
                    shifted_img[:, h-shift_h:, :] = 0
                    shifted_mask[h-shift_h:, :] = 0
                else:
                    shifted_img[:, shift_h:, :] = shifted_img[:, :h-shift_h, : ]
                    shifted_mask[shift_h:, :] = shifted_mask[:h-shift_h, : ]
                    shifted_img[:, :shift_h, :] = 0
                    shifted_mask[:shift_h, :] = 0
            return shifted_img, shifted_mask
        
        else:
            return img, mask
    
    def dropout(self, img: np.ndarray, mask: np.ndarray, shift_ratio: int, p: float):
        # Nếu dropout ngay giữa segment có mất đi thông tin về cấu trúc của object ko ?
        pass


if __name__ == '__main__':
    aug = Augmentation()
    img = cv2.imread('dataset/vis/train_0.jpg').transpose((-1, 0, 1))
    mask = np.load('dataset/train_masks/train_0.npy').argmax(-1)
    aug.translate(img, mask, 25, 0.5)