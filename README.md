# VapourSynth-SRMD-ncnn-Vulkan

SRMD super resolution for VapourSynth, based on [srmd-ncnn-vulkan](https://github.com/nihui/srmd-ncnn-vulkan). Some code is from [vapoursynth-waifu2x-ncnn-vulkan](https://github.com/Nlzy/vapoursynth-waifu2x-ncnn-vulkan).

## Usage

```python
core.srmdnv.SRMD(clip clip, [int scale, int noise, int tilesize,
                 int gpu_id, int gpu_thread, bool tta])
```

Models should be located in folder `models`, and folder `models` should be located in the same folder as dynamic link library.

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

* srmd-ncnn-vulkan: https://github.com/nihui/srmd-ncnn-vulkan
* vapoursynth-waifu2x-ncnn-vulkan: https://github.com/Nlzy/vapoursynth-waifu2x-ncnn-vulkan
