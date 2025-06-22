import cv2
import numpy as np
from tqdm import tqdm
import argparse
import warnings
import utils


def bkgr_unique(cap: cv2.VideoCapture, frame_index: int = 0) -> np.ndarray:
    """
    Return the frame at `frame_index` as the background.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, bkgr = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read frame at index {frame_index}.")
    return bkgr


def bkgr_median(cap: cv2.VideoCapture, max_frames: int, use_mean: bool = False) -> np.ndarray:
    """
    Compute pixel-wise median (or mean) across up to `max_frames` frames.
    """
    # guards memeory overload
    if max_frames > 128:
        warnings.warn("Reducing max_frames to 128")
        max_frames = 128
        
    frames = []
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    if not frames:
        raise RuntimeError("No frames read for background computation.")

    stack = np.stack(frames, axis=0)
    if use_mean:
        return np.mean(stack, axis=0).astype(np.uint8)
    else:
        return np.median(stack, axis=0).astype(np.uint8)
  
    
def bkgr_minim(cap: cv2.VideoCapture, max_frames: int) -> np.ndarray:
    """
    Aggregate the frame corresponding to minimum values for pixels
    """
    ret, bkgr = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")
    for _ in range(1, max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        bkgr = np.minimum(frame, bkgr)
    return bkgr


def bkgr_pmode(cap: cv2.VideoCapture, max_frames: int) -> np.ndarray:
    ret, bgr0 = cap.read()
    if not ret:
        return None
        
    H, W = bgr0.shape[:2]
    
    # Histogram for each pixel, each channel
    hist = np.zeros((H, W, 3, 256), dtype=np.uint16)
    
    def accumulate(frame):
        rows = np.arange(H)[:, None]
        cols = np.arange(W)[None, :]
        for ch in range(3):
            vals = frame[:, :, ch]
            np.add.at(hist[:, :, ch], (rows, cols, vals), 1)
    
    # Process first frame
    accumulate(bgr0)
    
    # Process remaining frames
    for _ in range(1, max_frames):
        ret, bgr = cap.read()
        if not ret:
            break
        accumulate(bgr)
       
    # Get mode for each pixel
    bkgr = hist.argmax(axis=-1).astype(np.uint8)
    
    return bkgr
        

class ExponentialBackground:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.background = None

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """
        Update background model with new frame, return current background.
        """
        frame = frame.astype(np.float32)

        if self.background is None:
            self.background = frame.copy()
        else:
            self.background = self.alpha * frame + (1 - self.alpha) * self.background

        return self.background.astype(np.uint8)

    def subtract(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute foreground = frame - current background
        """
        background = self.__call__(frame)
        foreground = cv2.absdiff(frame.astype(np.uint8), background)
        return foreground
    

# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Estimate video background with several methods.")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("--method", choices=["unique", "median", "mean", "minim", "pmode", "expo"],
                        default="median", help="Background estimation method")
    parser.add_argument("--frames", type=int, default=100,
                        help="Maximum number of frames to use where applicable")
    parser.add_argument("--index", type=int, default=0,
                        help="Frame index for the 'unique' method")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Exponential-average smoothing factor")
    parser.add_argument("--output", default="background.png",
                        help="Output image filename")
    parser.add_argument("--show", action="store_true",
                        help="Display the resulting background image")
    args = parser.parse_args()

    cap = utils.open_video(args.video)

    if args.method == "unique":
        background = bkgr_unique(cap, args.index)
    elif args.method == "median":
        background = bkgr_median(cap, args.frames, use_mean=False)
    elif args.method == "mean":
        background = bkgr_median(cap, args.frames, use_mean=True)
    elif args.method == "minim":
        background = bkgr_minim(cap, args.frames)
    elif args.method == "pmode":
        background = bkgr_pmode(cap, args.frames)
    elif args.method == "expo":
        exp_bg = ExponentialBackground(alpha=args.alpha)
        background = None
        for _ in range(args.frames):
            ret, frame = cap.read()
            if not ret:
                break
            background = exp_bg(frame)
        if background is None:
            raise RuntimeError("No frames processed for exponential background.")
    else:
        raise ValueError(f"Unknown method: {args.method}")

    cap.release()

    cv2.imwrite(args.output, background)
    if args.show:
        cv2.imshow("Background", background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(f"Background saved to {args.output}")

if __name__ == "__main__":
    main()