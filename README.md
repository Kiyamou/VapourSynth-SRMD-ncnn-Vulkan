# VapourSynth-SRMD-ncnn-Vulkan

SRMD super resolution for VapourSynth, based on [srmd-ncnn-vulkan](https://github.com/nihui/srmd-ncnn-vulkan).

## Usage

```python
core.jinc.JincResize(clip clip, [int scale, int noise, int tilesize,
                     int gpu_id, int gpu_thread, bool tta])
```

* ***clip***
    * Required parameter.
    * Clip to process.
    * Only 32bit RGB is supported.
* ***scale***
    * Optional parameter. Range: 2–4. *Default: 4*.
* ***noise***
    * Optional parameter. Range: -1–10. *Default: 3*.
    * The larger value, the stronger the effect of denoise.

## Reference Code

vapoursynth-waifu2x-ncnn-vulkan: https://github.com/Nlzy/vapoursynth-waifu2x-ncnn-vulkan