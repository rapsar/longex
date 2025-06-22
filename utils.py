import cv2
import numpy as np
import argparse
from pymediainfo import MediaInfo
import matplotlib.pyplot as plt
from PIL import Image, PngImagePlugin
from typing import Optional
from osxmetadata import OSXMetaData


def open_video(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    return cap


def get_num_frames(cap: cv2.VideoCapture) -> int:
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def show_and_save_image(img: np.ndarray, out_path: str) -> None:
    cv2.imshow('Image', img)
    cv2.imwrite(out_path, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def plot_brightness(brightness_values: list[float]) -> None:
    plt.plot(brightness_values)
    plt.xlabel('Frame')
    plt.ylabel('Mean Pixel Brightness')
    plt.title('Frame Brightness Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def metadata(video_path: str, print_output: bool = True) -> dict[str, str]:
    """
    Extract frame metadata (inaccessible by exif)
    Run:
    >> python utils.py metadata /path/to/video.MP4
    
    Alternatively from Terminal for all fields:
    >> mediainfo /path/to/video.MP4 
    """
    # The “canonical” keys you want to expose
    desired_keys = [
        "ISOSensitivity_FirstFrame",
        "ShutterSpeed_Time_FirstFrame",
        "CaptureFrameRate_FirstFrame",
        "IrisFNumber_FirstFrame",
        "FocusPositionFromImagePlane_FirstFrame"
    ]

    media = MediaInfo.parse(video_path)
    result: dict[str, str] = {}

    for track in media.tracks:
        td = track.to_data()  # a dict of whatever MediaInfo found
        for desired in desired_keys:
            # scan all the returned keys for a case-insensitive match
            for td_key, td_val in td.items():
                lk, ld = td_key.lower(), desired.lower()
                if lk == ld or ld in lk:
                    if td_val is not None:
                        result[desired] = td_val
                    break
    
    # print resul to consol
    if print_output:
        if not result:
            print(f"No first-frame metadata found in: {video_path}")
        else:
            print(f"First-frame metadata for {video_path}:")
            for key, value in result.items():
                print(f"{key:35s}: {value}")
        
    return result


def write_png_metadata(png_path: str, metadata: dict[str, str], out_path: Optional[str] = None) -> None:
    """
    Embed metadata into a PNG file as text chunks.

    Args:
        png_path: Path to the existing PNG image.
        metadata: Dictionary of metadata keys and values to embed.
        out_path: If provided, save to this path; otherwise overwrite png_path.
    """
    img = Image.open(png_path)
    meta = PngImagePlugin.PngInfo()
    for key, value in metadata.items():
        meta.add_text(str(key), str(value))
    # Determine save path
    save_path = out_path or png_path
    # Save with metadata
    img.save(save_path, "PNG", pnginfo=meta)


def write_metadata_as_finder_comment(png_path: str, metadata: dict[str, str]) -> None:
    """
    Store metadata so it shows up in Finder ▸ Get Info ▸ Comments.
    """
    comment = "; ".join(f"{k}={v}" for k, v in metadata.items())
    md = OSXMetaData(png_path)
    md.findercomment = comment        # writes .DS_Store + Spotlight attr


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="utils", description="Utility entry points.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sub-command: metadata
    parser_meta = subparsers.add_parser("metadata", help="Extract first-frame metadata")
    parser_meta.add_argument("video_path", help="Path to the video file")

    args = parser.parse_args()

    if args.command == "metadata":
        result = metadata(args.video_path, print_output = True)
    else:
        parser.print_help()