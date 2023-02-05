# VapourSynth-SRMD-ncnn-Vulkan

[![Build Status](https://github.com/Kiyamou/VapourSynth-SRMD-ncnn-Vulkan/workflows/CI/badge.svg)](https://github.com/Kiyamou/VapourSynth-SRMD-ncnn-Vulkan/actions)

SRMD super resolution for VapourSynth, based on [srmd-ncnn-vulkan](https://github.com/nihui/srmd-ncnn-vulkan). Some code is from [vapoursynth-waifu2x-ncnn-vulkan](https://github.com/Nlzy/vapoursynth-waifu2x-ncnn-vulkan).

## Usage

```python
core.srmdnv.SRMD(clip clip, [int scale, int noise, int tilesize_x, int tilesize_y,
                 int gpu_id, int gpu_thread, bool tta])
```

Models should be located in folder `models`, and folder `models` should be located in the same folder as dynamic link library.

* ***clip***
  * Required parameter.
  * Clip to process.
  * Only 32bit RGB is supported.
* ***scale***
  * Optional parameter. Range: 2–4. *Default: 2*.
* ***noise***
  * Optional parameter. Range: -1–10. *Default: 3*.
  * The larger value, the stronger the effect of denoise.
* ***tilesize_x***
  * Optional parameter. *Default: depands on video memory size*
  * The tilesize for horizontal.
  * Recommend to set a value that can divide the width.
* ***tilesize_y***
  * Optional parameter. *Default: same as tilesize_x*.
  * The tilesize for vertical.
  * Recommend to set a value that can divide the height.
* ***gpu_id***
  * Optional parameter. *Default: 0*.
  * If you have more than one gpu devices, you can select gpu device by the parameter.
* ***tta***
  * Optional parameter. *Default: False*.
  * If true, quality will be improved, but speed will be significantly slower.

## Compilation

### Windows

1.Install Vulkan SDK.

2.If your VapourSynth is installed in `C:\Program Files\VapourSynth` , you can run the following command directly. Otherwise use `cmake -G "NMake Makefiles" -DVAPOURSYNTH_INCLUDE_DIR=Path/To/vapoursynth/sdk/include/vapoursynth ..` in the second-to-last step.

```bash
git clone --recursive https://github.com/Kiyamou/VapourSynth-SRMD-ncnn-Vulkan.git
cd VapourSynth-SRMD-ncnn-Vulkan

mkdir build && cd build
cmake -G "NMake Makefiles" ..
cmake --build .
```

### Linux

1.Install Vulkan SDK and add to path.

2.If your VapourSynth is installed in `usr/local` , you can run the following command directly. Otherwise use `cmake -DVAPOURSYNTH_INCLUDE_DIR=Path/To/vapoursynth ..` in the second-to-last step.

```bash
git clone --recursive https://github.com/Kiyamou/VapourSynth-SRMD-ncnn-Vulkan.git
cd VapourSynth-SRMD-ncnn-Vulkan

mkdir build && cd build
cmake ..
cmake --build .
```

### Windows and Linux using Github Actions

1.[Fork this repository](https://github.com/Kiyamou/VapourSynth-SRMD-ncnn-Vulkan/fork).

2.Enable Github Actions on your fork: **Settings** tab -> **Actions** -> **General** -> **Allow all actions and reusable workflows** -> **Save** button.

3.Edit (if necessary) the file `.github/workflows/CI.yml` on your fork modifying the environment variables Vulkan SDK and/or VapourSynth versions:

```
env:
  VULKAN_SDK_VERSION: <SET_YOUR_VERSION>
  VAPOURSYNTH_VERSION: <SET_YOUR_VERSION>
```

4.Go to the GitHub **Actions** tab on your fork, select **CI** workflow and press the **Run workflow** button (if you modified the `.github/workflows/CI.yml` file, a workflow will be already running and no need to run a new one).

When the workflow is completed you will be able to download the artifacts generated (Windows and Linux versions) from the run.

## Download Nightly Builds

**GitHub Actions Artifacts ONLY can be downloaded by GitHub logged users.**

Nightly builds are built automatically by GitHub Actions (GitHub's integrated CI/CD tool) every time a new commit is pushed to the _master_ branch or a pull request is created.

To download the latest nightly build, go to the GitHub [Actions](https://github.com/Kiyamou/VapourSynth-SRMD-ncnn-Vulkan/actions/workflows/CI.yml) tab, enter the last run of workflow **CI**, and download the artifacts generated (Windows and Linux versions) from the run.

## Reference Code

* srmd-ncnn-vulkan: https://github.com/nihui/srmd-ncnn-vulkan
* vapoursynth-waifu2x-ncnn-vulkan: https://github.com/Nlzy/vapoursynth-waifu2x-ncnn-vulkan
