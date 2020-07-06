// srmd implemented with ncnn library

#include <map>
#include <memory>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <fstream>

#include "srmd.h"

#include "VapourSynth.h"
#include "VSHelper.h"

class Semaphore
{
private:
    int val;
    std::mutex mtx;
    std::condition_variable cv;

public:
    explicit Semaphore(int init_value) : val(init_value) {}

    void wait()
    {
        std::unique_lock<std::mutex> lock(mtx);
        while (val <= 0)
        {
            cv.wait(lock);
        }
        val--;
    }

    void signal()
    {
        std::lock_guard<std::mutex> guard(mtx);
        val++;
        cv.notify_one();
    }
};

struct FilterData
{
    VSNodeRef* node;
    const VSVideoInfo* vi;
    int target_width, target_height;
    SRMD* srmd;
    Semaphore* gpuSemaphore;
};

static std::mutex g_lock{};
static int g_filter_instance_count = 0;
static std::map<int, Semaphore*> g_gpu_semaphore;

static void process(const VSFrameRef* src, VSFrameRef* dst, const FilterData* const VS_RESTRICT d, const VSAPI* vsapi) noexcept
{
    if (d->vi->format->colorFamily == cmRGB)
    {
        int src_width = vsapi->getFrameWidth(src, 0);
        int src_height = vsapi->getFrameHeight(src, 0);
        int src_stirde = vsapi->getStride(src, 0) / sizeof(float);
        int dst_stride = vsapi->getStride(dst, 0) / sizeof(float);

        const float* srcpR = reinterpret_cast<const float*>(vsapi->getReadPtr(src, 0));
        const float* srcpG = reinterpret_cast<const float*>(vsapi->getReadPtr(src, 1));
        const float* srcpB = reinterpret_cast<const float*>(vsapi->getReadPtr(src, 2));

        float* VS_RESTRICT dstpR = reinterpret_cast<float*>(vsapi->getWritePtr(dst, 0));
        float* VS_RESTRICT dstpG = reinterpret_cast<float*>(vsapi->getWritePtr(dst, 1));
        float* VS_RESTRICT dstpB = reinterpret_cast<float*>(vsapi->getWritePtr(dst, 2));

        d->gpuSemaphore->wait();
        d->srmd->process(srcpR, srcpG, srcpB, dstpR, dstpG, dstpB, src_width, src_height, 3, src_stirde, dst_stride);
        d->gpuSemaphore->signal();
    }
}

static void VS_CC filterInit(VSMap* in, VSMap* out, void** instanceData, VSNode* node, VSCore* core, const VSAPI* vsapi)
{
    FilterData* d = static_cast<FilterData*>(*instanceData);
    VSVideoInfo dst_vi = (VSVideoInfo)* (d->vi);
    dst_vi.width = d->target_width;
    dst_vi.height = d->target_height;
    vsapi->setVideoInfo(&dst_vi, 1, node);
}

static const VSFrameRef* VS_CC filterGetFrame(int n, int activationReason, void** instancData, void** frameData, VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi)
{
    const FilterData* d = static_cast<const FilterData*>(*instancData);

    if (activationReason == arInitial)
    {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    }
    else if (activationReason == arAllFramesReady)
    {
        const VSFrameRef* src = vsapi->getFrameFilter(n, d->node, frameCtx);
        VSFrameRef* dst = vsapi->newVideoFrame(d->vi->format, d->target_width, d->target_height, src, core);

        process(src, dst, d, vsapi);

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC filterFree(void* instanceData, VSCore* core, const VSAPI* vsapi)
{
    FilterData* d = static_cast<FilterData*>(instanceData);
    vsapi->freeNode(d->node);

    delete d->srmd;
    delete d;

    std::lock_guard<std::mutex> guard(g_lock);
    g_filter_instance_count--;
    if (g_filter_instance_count == 0)
    {
        ncnn::destroy_gpu_instance();
        for (auto pair : g_gpu_semaphore)
        {
            delete pair.second;
        }
        g_gpu_semaphore.clear();
    }
}

static void VS_CC filterCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi)
{
    std::unique_ptr<FilterData> d = std::make_unique<FilterData>();
    int err;

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    {
        std::lock_guard<std::mutex> guard(g_lock);

        if (g_filter_instance_count == 0)
            ncnn::create_gpu_instance();

        g_filter_instance_count++;
    }

    try
    {
        if (!isConstantFormat(d->vi) ||
            d->vi->format->sampleType == stInteger ||
            (d->vi->format->sampleType == stFloat && d->vi->format->bitsPerSample != 32))
            throw std::string{ "only constant format 32 bits float input supported" };

        int scale = int64ToIntS(vsapi->propGetInt(in, "scale", 0, &err));
        if (err)
            scale = 2;
        if (scale < 2 || scale > 4)
            throw std::string{ "model is only supported 2 - 4 scale" };

        d->target_width = d->vi->width * scale;
        d->target_height = d->vi->height * scale;

        int noise = int64ToIntS(vsapi->propGetInt(in, "noise", 0, &err));
        if (err)
            noise = 3;
        if (noise < -1 || noise > 10)
            throw std::string{ "noise mode must be within -1 - 10" };

        // Model path
        const std::string pluginPath{ vsapi->getPluginPath(vsapi->getPluginById("com.vapoursynth.srmdnv", core)) };
        std::string paramPath{ pluginPath.substr(0, pluginPath.find_last_of('/')) };
        std::string modelPath{ pluginPath.substr(0, pluginPath.find_last_of('/')) };
        if (noise == -1)
        {
            paramPath += "/models/models-srmd/srmdnf_x" + std::to_string(scale) + ".param";
            modelPath += "/models/models-srmd/srmdnf_x" + std::to_string(scale) + ".bin";
        }
        else
        {
            paramPath += "/models/models-srmd/srmd_x" + std::to_string(scale) + ".param";
            modelPath += "/models/models-srmd/srmd_x" + std::to_string(scale) + ".bin";
        }

        // Check model file readable
        std::ifstream pf(paramPath);
        std::ifstream mf(modelPath);
        if (!pf.good() || !mf.good())
            throw std::string{ "can't open model file" };

        // GPU id
        int gpuId = int64ToIntS(vsapi->propGetInt(in, "gpu_id", 0, &err));
        if (err)
            gpuId = 0;
        if (gpuId < 0 || gpuId >= ncnn::get_gpu_count())
			throw std::string{ "invalid 'gpu_id'" };

        // Tile size
        int tilesize_x = int64ToIntS(vsapi->propGetInt(in, "tilesize_x", 0, &err));
        if (err)
            tilesize_x = 0;
        if (tilesize_x != 0 && tilesize_x < 32)
            throw std::string{ "block_w must be >= 32 or set as 0" };

        int tilesize_y = int64ToIntS(vsapi->propGetInt(in, "tilesize_y", 0, &err));
        if (err)
            tilesize_y = tilesize_x;
        if (tilesize_y != 0 && tilesize_y < 32)
            throw std::string{ "block_h must be >= 32 or set as 0" };

        // More fine-grained tilesize policy here
        uint32_t heap_budget = ncnn::get_gpu_device(gpuId)->get_heap_budget();
        if (tilesize_x == 0)
        {
            if (heap_budget > 2600)
                tilesize_x = 400;
            else if (heap_budget > 740)
                tilesize_x = 200;
            else if (heap_budget > 250)
                tilesize_x = 100;
            else
                tilesize_x = 32;
        }
        if (tilesize_y == 0)
        {
            if (heap_budget > 2600)
                tilesize_y = 400;
            else if (heap_budget > 740)
                tilesize_y = 200;
            else if (heap_budget > 250)
                tilesize_y = 100;
            else
                tilesize_y = 32;
        }

        int gpuThread;
        int customGpuThread = int64ToIntS(vsapi->propGetInt(in, "gpu_thread", 0, &err));
        if (customGpuThread > 0)
            gpuThread = customGpuThread;
        else
            gpuThread = int64ToIntS(ncnn::get_gpu_info(gpuId).transfer_queue_count);
        gpuThread = std::min(gpuThread, int64ToIntS(ncnn::get_gpu_info(gpuId).compute_queue_count));

        std::lock_guard<std::mutex> guard(g_lock);
        if (!g_gpu_semaphore.count(gpuId))
            g_gpu_semaphore.insert(std::pair<int, Semaphore*>(gpuId, new Semaphore(gpuThread)));
        d->gpuSemaphore = g_gpu_semaphore.at(gpuId);

        bool tta = !!vsapi->propGetInt(in, "tta", 0, &err);

        d->srmd = new SRMD(gpuId, tta);
        d->srmd->scale = scale;
        d->srmd->noise = noise;
        d->srmd->tilesize_x = tilesize_x;
        d->srmd->tilesize_y = tilesize_y;
        d->srmd->prepadding = 12;
        d->srmd->load(paramPath, modelPath);
    }
    catch (const std::string &error)
    {
        {
            std::lock_guard<std::mutex> guard(g_lock);

            g_filter_instance_count--;
            if (g_filter_instance_count == 0)
                ncnn::destroy_gpu_instance();
        }

        vsapi->setError(out, ("SRMD: " + error).c_str());
        vsapi->freeNode(d->node);
        return;
    }

    vsapi->createFilter(in, out, "SRMD", filterInit, filterGetFrame, filterFree, fmParallel, 0, d.release(), core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin* plugin)
{
    configFunc("com.vapoursynth.srmdnv", "srmdnv", "SRMD ncnn Vulkan plugin", VAPOURSYNTH_API_VERSION, 1, plugin);

    registerFunc("SRMD",
                 "clip:clip;"
                 "scale:int:opt;"
                 "noise:int:opt;"
                 "tilesize_x:int:opt;"
                 "tilesize_y:int:opt;"
                 "gpu_id:int:opt;"
                 "gpu_thread:int:opt;"
                 "tta:int:opt",
                 filterCreate, 0, plugin);
}
