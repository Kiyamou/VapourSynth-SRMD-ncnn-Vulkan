// srmd implemented with ncnn library

#ifndef SRMD_H
#define SRMD_H

#include <string>

// ncnn
#include "net.h"
#include "gpu.h"
#include "layer.h"

class SRMD
{
public:
    SRMD(int gpuid, bool _tta_mode = false);
    ~SRMD();

    int load(const std::string& parampath, const std::string& modelpath);
    int process(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int width, int height, int channels, int src_stride, int dst_stride) const;

public:
    // srmd parameters
    int scale;
    int noise;
    int tilesize_x;
    int tilesize_y;
    int prepadding;

private:
    ncnn::VulkanDevice* vkdev;
    ncnn::Net net;
    ncnn::Pipeline* srmd_preproc;
    ncnn::Pipeline* srmd_postproc;
    bool tta_mode;
};

#endif // SRMD_H
