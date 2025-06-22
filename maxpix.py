"""
maxpix.py

Create a long-exposure image by computing the maximum pixel value across all frames of a video.
Also records and plots the mean brightness of each frame.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import utils
from tqdm import tqdm
from pathlib import Path
import argparse
import os
 
    
def read_first_frame(cap: cv2.VideoCapture) -> np.ndarray:
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Cannot read first frame.")    
    return frame.astype(np.uint8)


def build_max_frame(
    cap: cv2.VideoCapture,
    n_frames: int,
    channel: str,
    on_frame: callable = None,
    sample_every: int = 1
) -> tuple[np.ndarray, list[float]]:
    
    max_frame = read_first_frame(cap)

    brightness_values = [max_frame.mean()]
    channel_map = {'B': 0, 'G': 1, 'R': 2}
    ch = channel_map.get(channel.upper(), 1)  # default to G

    for frame_idx in tqdm(range(1, n_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        brightness_values.append(frame.mean())

        update_mask = frame[:, :, ch] > max_frame[:, :, ch]
        rows, cols = np.where(update_mask)
        max_frame[rows, cols] = frame[rows, cols]

        # callback to stream frame (e.g. write to GIF)
        if on_frame and frame_idx % sample_every == 0:
            on_frame(max_frame.copy(), frame_idx)

    return max_frame, brightness_values


def build_max_frame_with_gif(
    cap: cv2.VideoCapture,
    n_frames: int,
    channel: str,
    gif_path: str,
    sample_every: int = 1,
    duration: float = 0.05
) -> tuple[np.ndarray, list[float]]:
    with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:

        def stream_to_gif(frame: np.ndarray, idx: int):
            frame = cv2.resize(frame, (640, 360))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)

        return build_max_frame(
            cap,
            n_frames,
            channel,
            on_frame=stream_to_gif,
            sample_every=sample_every
        )
    
    
def main():
    parser = argparse.ArgumentParser(
        description="Create a long-exposure image (and optional GIF) by max-pixel projection."
    )
    parser.add_argument(
        "-v", "--video", required=True, help="Path to input video file."
    )
    parser.add_argument(
        "-c", "--channel", choices=['B','G','R'], default='G',
        help="Color channel to drive the max projection (default: G)."
    )
    parser.add_argument(
        "--gif", action="store_true", default=False,
        help="Whether to output an animated GIF."
    )
    parser.add_argument(
        "--gif_path", default=None,
        help="Path to output GIF file (required if --gif)."
    )
    parser.add_argument(
        "--sample_every", type=int, default=1,
        help="Frame sampling interval for GIF generation."
    )
    parser.add_argument(
        "--duration", type=float, default=0.05,
        help="Frame duration for GIF (in seconds)."
    )
    parser.add_argument(
        "-n", "--n_frames", type=int, default=None,
        help="Number of frames to process (default: all frames in the video)."
    )
    parser.add_argument(
        "-o", "--out_dir", default="res",
        help="Output directory for images and GIF."
    )
    args = parser.parse_args()

    video_path = args.video
    filename = Path(video_path).stem
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    cap = utils.open_video(video_path)
    if args.n_frames is None:
        n_frames = utils.get_num_frames(cap)
    else:
        n_frames = args.n_frames

    if args.gif:
        gif_path = args.gif_path or os.path.join(out_dir, f"{filename}.gif")
        max_frame, brightness_values = build_max_frame_with_gif(
            cap,
            n_frames,
            channel=args.channel,
            gif_path=gif_path,
            sample_every=args.sample_every,
            duration=args.duration
        )
    else:
        max_frame, brightness_values = build_max_frame(
            cap,
            n_frames,
            channel=args.channel
        )

    # Plot brightness and save
    utils.plot_brightness(brightness_values)
    img_path = os.path.join(out_dir, f"maxpix_{filename}.png")
    utils.show_and_save_image(img=max_frame, out_path=img_path)
    print(f"Saved composite image to {img_path}")
    if args.gif:
        print(f"Saved animated GIF to {gif_path}")
        
    cap.release()
    
    # display movie metadata
    video_metadata = utils.metadata(video_path, print_output=True)
    # add to output png
    utils.write_metadata_as_finder_comment(img_path, video_metadata)

if __name__ == "__main__":
    main()