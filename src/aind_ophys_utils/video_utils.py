""" Utils for video generation """
from pathlib import Path

import h5py
import imageio_ffmpeg as mpg
import numpy as np

from aind_ophys_utils.array_utils import downsample_array


def downsample_h5_video(
    video_path: Path,
    input_fps: float = 31.0,
    output_fps: float = 4.0,
    strategy: str = "mean",
) -> np.ndarray:
    """Opens an h5 file and downsamples dataset 'data'
    along axis=0

    Parameters
    ----------
        video_path: pathlib.Path
            path to an h5 video. Should have dataset 'data'. For video,
            assumes dimensions [time, height, width] and downsampling
            applies to time.
        input_fps: float
            frames-per-second of the input array
        output_fps: float
            frames-per-second of the output array
        strategy: str
            downsampling strategy. 'max', 'mean', 'median',
            'first', 'last', 'mid'.

    Returns:
        video_out: numpy.ndarray
            array downsampled along axis=0
    """
    with h5py.File(video_path, "r") as h5f:
        video_out = downsample_array(
            h5f["data"], input_fps, output_fps, strategy=strategy
        )
    return video_out


def encode_video(
    video: np.ndarray,
    output_path: str,
    fps: float,
    bitrate: str = "0",
    crf: int = 20,
    cpu_used: int = 4,
    color: bool = False,
) -> str:
    """Encode a video with vp9 codec via imageio-ffmpeg

    Parameters
    ----------
    video : np.ndarray
        Video to be encoded
    output_path : str
        Desired output path for encoded video
    fps : float
        Desired frame rate for encoded video
    bitrate : str, optional
        Desired bitrate of output, by default "0". The default *MUST*
        be zero in order to encode in constant quality mode. Other values
        will result in constrained quality mode.
    crf : int, optional
        Desired perceptual quality of output, by default 20. Value can
        be from 0 - 63. Lower values mean better quality (but bigger video
        sizes).
    cpu_used : int, optional
        Sets how efficient the compression will be, by default 4. Values can
        be between 0 and 5. Higher values increase encoding speed at the
        expense of having some impact on quality and rate control accuracy.
    color : bool, optional
        Whether the input video is color or grayscale, by default False

    Returns
    -------
    str
        Output path of the encoded video
    """

    # ffmpeg expects video shape in terms of: (width, height)
    video_shape = (video[0].shape[1], video[0].shape[0])

    writer = mpg.write_frames(
        output_path,
        video_shape,
        pix_fmt_in="rgb24" if color else "gray8",
        pix_fmt_out="yuv420p",
        codec="libvpx-vp9",
        fps=fps,
        bitrate=bitrate,
        output_params=[
            "-crf",
            str(crf),
            "-row-mt",
            "1",
            "-cpu-used",
            str(cpu_used),
        ],
    )

    writer.send(None)  # Seed ffmpeg-imageio writer generator
    for frame in video:
        writer.send(frame)
    writer.close()

    return output_path
