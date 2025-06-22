import cv2
import numpy as np
import utils
from tqdm import tqdm
from skimage import exposure


def frame_to_frgr(
    frame: np.ndarray, 
    bkgr: np.ndarray, 
    delta: int, 
    channel: int = 1, 
    dilate_kernel: int = 0
    ) -> np.ndarray:
    """
    Extract foreground from a single frame using background subtraction.
    
    Args:
        frame: Current frame (H, W, C)
        background: Background image (H, W, C) 
        delta: Threshold for foreground detection
        channel: Channel to use for thresholding (0=B, 1=G, 2=R)
        dilate_kernel: Size of dilation kernel (0 = no dilation)
        
    Returns:
        tuple: (foreground_frame, mask)
            - foreground_frame: Frame with only foreground pixels, background=0
            - mask: Binary mask (H, W) indicating foreground pixels
    """
    # Ensure matching dtypes
    bkgr = bkgr.astype(frame.dtype)
    
    bkgr = exposure.match_histograms(bkgr, frame, channel_axis = 2)
    # Compute difference and create mask
    diff = cv2.subtract(frame, bkgr)
    diff_ch = diff[:, :, channel] if diff.ndim == 3 else diff
    mask = diff_ch > delta
    
    # Optional dilation
    if dilate_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel, dilate_kernel))
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    
    # Extract foreground pixels only
    mask_3c = mask[..., None]
    frgr_frame = np.where(mask_3c, frame, 0)
    
    return frgr_frame.astype(frame.dtype)


# ============================================================================
# MERGING STRATEGIES
# ============================================================================

def merge_latest(composite: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    """
    Always use the latest foreground pixels (simple replacement).
    """
    mask = np.any(foreground > 0, axis=2)
    mask_3c = mask[..., None]
    return np.where(mask_3c, foreground, composite)


def merge_maximum_pixelwise(composite: np.ndarray, foreground: np.ndarray, use_luminance: bool = False) -> np.ndarray:
    """
    Choose the RGB triplet with higher total intensity (preserves natural colors).
    """
    mask = np.any(foreground > 0, axis=2)
    
    # Perceptual luminance (assuming BGR format)
    def get_luminance(img):
        return 0.114 * img[:,:,0] + 0.587 * img[:,:,1] + 0.299 * img[:,:,2]
    
    # Calculate total intensity for each pixel
    if use_luminance:
        composite_intensity = get_luminance(composite)
        foreground_intensity = get_luminance(foreground)
    else:
        composite_intensity = np.sum(composite, axis=2)
        foreground_intensity = np.sum(foreground, axis=2)
    
    # Use foreground where it's brighter AND in the mask
    use_foreground = (foreground_intensity > composite_intensity) & mask
    use_foreground_3c = use_foreground[..., None]
    
    return np.where(use_foreground_3c, foreground, composite)


def merge_accumulation(
    composite: np.ndarray,
    foreground: np.ndarray,
    background: np.ndarray,
    method: str = "log",
    decay_factor: float = 0.95
) -> np.ndarray:
    """
    Combined accumulation merge. Two modes:
      - "log": logarithmic accumulation to compress bright values.
      - "weighted": weighted accumulation with decay; recent flashes more prominent.

    Args:
        composite: Current composite image (H, W, C) uint8.
        foreground: Current foreground image (H, W, C) uint8.
        background: Background image (H, W, C) uint8.
        method: "log" or "weighted" (default "log").
        decay_factor: Decay factor for weighted mode (default 0.95).

    Returns:
        np.ndarray: Updated composite image, same shape and dtype as inputs.
    """
    # mask of where there is any foreground
    mask = np.any(foreground > 0, axis=2)
    mask_3c = mask[..., None]

    # float buffers
    comp_f = composite.astype(np.float32)
    fg_f   = foreground.astype(np.float32)
    bg_f   = background.astype(np.float32)

    # positive-only firefly brightness
    contribution = np.maximum(fg_f - bg_f, 0)

    if method == "log":
        # logarithmic compression
        contrib = np.log1p(contribution)
        comp_f = comp_f + np.where(mask_3c, contrib, 0)
    elif method == "weighted":
        # decay + linear add
        comp_f = comp_f * decay_factor
        comp_f = comp_f + np.where(mask_3c, contribution, 0)
    else:
        raise ValueError(f"Unknown accumulation method: {method}")

    return np.clip(comp_f, 0, 255).astype(np.uint8)


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def frgr_static(
    cap: cv2.VideoCapture, 
    background: np.ndarray,
    max_frames: int,
    delta: int, 
    channel: int, 
    dilate_kernel: int,
    merge_method: str, 
    **merge_kwargs
    ) -> np.ndarray:
    """
    Process entire video using the specified merging strategy.
    """
    # Reset and read first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("No frames to process")
    composite = np.zeros_like(frame)

    # Determine total number of frames for progress bar
    if max_frames is None:
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Define merge function dispatch
    merge_functions = {
        'latest': merge_latest,
        'max_pixel': merge_maximum_pixelwise,
        'accum': merge_accumulation
    }
    if merge_method not in merge_functions:
        raise ValueError(f"Unknown merge method: {merge_method}")
    merge_func = merge_functions[merge_method]

    # Rewind and process frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for _ in tqdm(range(1, max_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Extract foreground and mask
        frgr_frame = frame_to_frgr(frame, background, delta, channel, dilate_kernel)

        # Merge according to method
        if merge_method == 'latest':
            composite = merge_func(composite, frgr_frame)
        elif merge_method == 'max_pixel':
            use_lum = merge_kwargs.get('use_luminance', False)
            composite = merge_func(composite, frgr_frame, use_luminance=use_lum)
        else:  # 'accum'
            composite = merge_func(composite, frgr_frame, background, method=merge_method, **merge_kwargs)

    return composite



if __name__ == "__main__":
    # Simple usage without argparse
    video_path = "/Volumes/Untitled/PRIVATE/M4ROOT/CLIP/20250604_RS0053.MP4"
    background_path = "res/minim_bkgr.png"
    
    delta = 11
    max_frames = 1000
    channel = 1
    dilate_kernel = 2
    merge_method = "log"

    # Open video and background
    cap = utils.open_video(video_path)
    background = cv2.imread(background_path)
    if background is None:
        raise FileNotFoundError(f"Background image not found: {background_path}")

    # Run foreground extraction
    composite = frgr_static(
        cap=cap,
        background=background,
        max_frames=max_frames,
        delta=delta,
        channel=channel,
        dilate_kernel=dilate_kernel,
        merge_method=merge_method
    )

    # Save result
    output_file = f"res/foreground_{merge_method}_delta_{delta}_kernel_{dilate_kernel}.png"
    cv2.imwrite(output_file, composite)
    print(f"Saved composite image to {output_file}")

    cap.release()