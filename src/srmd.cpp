// srmd implemented with ncnn library

#include "srmd.hpp"

#include <algorithm>
#include <vector>

#include "srmd_preproc.comp.hex.h"
#include "srmd_postproc.comp.hex.h"
#include "srmd_preproc_tta.comp.hex.h"
#include "srmd_postproc_tta.comp.hex.h"

SRMD::SRMD(int gpuid, bool _tta_mode)
{
    vkdev = gpuid == -1 ? 0 : ncnn::get_gpu_device(gpuid);

    srmd_preproc = 0;
    srmd_postproc = 0;
    tta_mode = _tta_mode;
}

SRMD::~SRMD()
{
    // cleanup preprocess and postprocess pipeline
    delete srmd_preproc;
    delete srmd_postproc;
}

int SRMD::load(const std::string& parampath, const std::string& modelpath)
{
    net.opt.use_vulkan_compute = true;
    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_int8_storage = false;
    net.opt.use_int8_arithmetic = false;

    net.set_vulkan_device(vkdev);
    
    net.load_param(parampath.c_str());
    net.load_model(modelpath.c_str());

    // initialize preprocess and postprocess pipeline
    if (vkdev)
    {
        std::vector<ncnn::vk_specialization_type> specializations(1);
#if _WIN32
        specializations[0].i = 1;
#else
        specializations[0].i = 0;
#endif

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    if (tta_mode)
                        compile_spirv_module(srmd_preproc_tta_comp_data, sizeof(srmd_preproc_tta_comp_data), net.opt, spirv);
                    else
                        compile_spirv_module(srmd_preproc_comp_data, sizeof(srmd_preproc_comp_data), net.opt, spirv);
                }
            }

            srmd_preproc = new ncnn::Pipeline(vkdev);
            srmd_preproc->set_optimal_local_size_xyz(8, 8, 3);
            srmd_preproc->create(spirv.data(), spirv.size() * 4, specializations);
        }

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            {
                ncnn::MutexLockGuard guard(lock);
                if (spirv.empty())
                {
                    if (tta_mode)
                        compile_spirv_module(srmd_postproc_tta_comp_data, sizeof(srmd_postproc_tta_comp_data), net.opt, spirv);
                    else
                        compile_spirv_module(srmd_postproc_comp_data, sizeof(srmd_postproc_comp_data), net.opt, spirv);
                }
            }

            srmd_postproc = new ncnn::Pipeline(vkdev);
            srmd_postproc->set_optimal_local_size_xyz(8, 8, 3);
            srmd_postproc->create(spirv.data(), spirv.size() * 4, specializations);
        }
    }

    return 0;
}

int SRMD::process(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int width, int height, int src_stride, int dst_stride) const
{
    const int TILE_SIZE_X = tilesize_x;
    const int TILE_SIZE_Y = tilesize_y;

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    // each tile 400x400
    const int xtiles = (width + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (height + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    //#pragma omp parallel for num_threads(2)
    for (int yi = 0; yi < ytiles; yi++)
    {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, height) - yi * TILE_SIZE_Y;

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding, height);
        const int in_tile_w = width;
        const int in_tile_h = in_tile_y1 - in_tile_y0;

        ncnn::Mat in;
        in.create(in_tile_w, in_tile_h, CHANNELS, sizeof(float));

        float* in_tile_r = in.channel(0);
        float* in_tile_g = in.channel(1);
        float* in_tile_b = in.channel(2);
        
        const float* sr = srcpR + in_tile_y0 * src_stride;
        const float* sg = srcpG + in_tile_y0 * src_stride;
        const float* sb = srcpB + in_tile_y0 * src_stride;
        
        for (int y = 0; y < in_tile_h; y++)
        {
            for (int x = 0; x < in_tile_w; x++)
            {
                in_tile_r[in_tile_w * y + x] = sr[src_stride * y + x] * 255.f;
                in_tile_g[in_tile_w * y + x] = sg[src_stride * y + x] * 255.f;
                in_tile_b[in_tile_w * y + x] = sb[src_stride * y + x] * 255.f;
            }
        }

        ncnn::VkCompute cmd(vkdev);

        // upload
        ncnn::VkMat in_gpu;
        {
            cmd.record_clone(in, in_gpu, opt);

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
        int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, height);

        ncnn::VkMat out_gpu;
        out_gpu.create(width * scale, (out_tile_y1 - out_tile_y0) * scale, CHANNELS, sizeof(float), blob_vkallocator);

        for (int xi = 0; xi < xtiles; xi++)
        {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, width) - xi * TILE_SIZE_X;

            if (tta_mode)
            {
                // preproc
                ncnn::VkMat in_tile_gpu[8];
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, width) + prepadding;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, height) + prepadding;

                    in_tile_gpu[0].create(tile_x1 - tile_x0, tile_y1 - tile_y0, noise == -1 ? 18 : 19, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[1].create(tile_x1 - tile_x0, tile_y1 - tile_y0, noise == -1 ? 18 : 19, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[2].create(tile_x1 - tile_x0, tile_y1 - tile_y0, noise == -1 ? 18 : 19, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[3].create(tile_x1 - tile_x0, tile_y1 - tile_y0, noise == -1 ? 18 : 19, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[4].create(tile_y1 - tile_y0, tile_x1 - tile_x0, noise == -1 ? 18 : 19, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[5].create(tile_y1 - tile_y0, tile_x1 - tile_x0, noise == -1 ? 18 : 19, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[6].create(tile_y1 - tile_y0, tile_x1 - tile_x0, noise == -1 ? 18 : 19, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[7].create(tile_y1 - tile_y0, tile_x1 - tile_x0, noise == -1 ? 18 : 19, in_out_tile_elemsize, 1, blob_vkallocator);

                    std::vector<ncnn::VkMat> bindings(9);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu[0];
                    bindings[2] = in_tile_gpu[1];
                    bindings[3] = in_tile_gpu[2];
                    bindings[4] = in_tile_gpu[3];
                    bindings[5] = in_tile_gpu[4];
                    bindings[6] = in_tile_gpu[5];
                    bindings[7] = in_tile_gpu[6];
                    bindings[8] = in_tile_gpu[7];

                    std::vector<ncnn::vk_constant_type> constants(12);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu[0].w;
                    constants[4].i = in_tile_gpu[0].h;
                    constants[5].i = in_tile_gpu[0].cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = noise;
                    constants[11].i = CHANNELS;  // (noise == -1 ? 18 : 19) + CHANNELS - 3;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu[0].w;
                    dispatcher.h = in_tile_gpu[0].h;
                    dispatcher.c = (noise == -1 ? 18 : 19) + CHANNELS - 3;

                    cmd.record_pipeline(srmd_preproc, bindings, constants, dispatcher);
                }

                // srmd
                ncnn::VkMat out_tile_gpu[8];
                for (int ti = 0; ti < 8; ti++)
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(blob_vkallocator);
                    ex.set_workspace_vkallocator(blob_vkallocator);
                    ex.set_staging_vkallocator(staging_vkallocator);

                    ex.input("input", in_tile_gpu[ti]);

                    ex.extract("output", out_tile_gpu[ti], cmd);
                }

                // postproc
                {
                    std::vector<ncnn::VkMat> bindings(9);
                    bindings[0] = out_tile_gpu[0];
                    bindings[1] = out_tile_gpu[1];
                    bindings[2] = out_tile_gpu[2];
                    bindings[3] = out_tile_gpu[3];
                    bindings[4] = out_tile_gpu[4];
                    bindings[5] = out_tile_gpu[5];
                    bindings[6] = out_tile_gpu[6];
                    bindings[7] = out_tile_gpu[7];
                    bindings[8] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(11);
                    constants[0].i = out_tile_gpu[0].w;
                    constants[1].i = out_tile_gpu[0].h;
                    constants[2].i = out_tile_gpu[0].cstep;
                    constants[3].i = out_gpu.w;
                    constants[4].i = out_gpu.h;
                    constants[5].i = out_gpu.cstep;
                    constants[6].i = xi * TILE_SIZE_X * scale;
                    constants[7].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[8].i = prepadding * scale;
                    constants[9].i = prepadding * scale;
                    constants[10].i = CHANNELS;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = CHANNELS;

                    cmd.record_pipeline(srmd_postproc, bindings, constants, dispatcher);
                }
            }
            else
            {
                // preproc
                ncnn::VkMat in_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, width) + prepadding;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, height) + prepadding;

                    in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, noise == -1 ? 18 : 19, in_out_tile_elemsize, 1, blob_vkallocator);

                    std::vector<ncnn::VkMat> bindings(2);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(12);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu.w;
                    constants[4].i = in_tile_gpu.h;
                    constants[5].i = in_tile_gpu.cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = noise;
                    constants[11].i = CHANNELS;  // (noise == -1 ? 18 : 19) + CHANNELS - 3;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu.w;
                    dispatcher.h = in_tile_gpu.h;
                    dispatcher.c = (noise == -1 ? 18 : 19) + CHANNELS - 3;

                    cmd.record_pipeline(srmd_preproc, bindings, constants, dispatcher);
                }

                // srmd
                ncnn::VkMat out_tile_gpu;
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(blob_vkallocator);
                    ex.set_workspace_vkallocator(blob_vkallocator);
                    ex.set_staging_vkallocator(staging_vkallocator);

                    ex.input("input", in_tile_gpu);

                    ex.extract("output", out_tile_gpu, cmd);
                }

                // postproc
                {
                    std::vector<ncnn::VkMat> bindings(2);
                    bindings[0] = out_tile_gpu;
                    bindings[1] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(11);
                    constants[0].i = out_tile_gpu.w;
                    constants[1].i = out_tile_gpu.h;
                    constants[2].i = out_tile_gpu.cstep;
                    constants[3].i = out_gpu.w;
                    constants[4].i = out_gpu.h;
                    constants[5].i = out_gpu.cstep;
                    constants[6].i = xi * TILE_SIZE_X * scale;
                    constants[7].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[8].i = prepadding * scale;
                    constants[9].i = prepadding * scale;
                    constants[10].i = CHANNELS;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = CHANNELS;

                    cmd.record_pipeline(srmd_postproc, bindings, constants, dispatcher);
                }
            }

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        // download
        {
            ncnn::Mat out;

            cmd.record_clone(out_gpu, out, opt);

            cmd.submit_and_wait();

            const float* out_tile_r = out.channel(0);
            const float* out_tile_g = out.channel(1);
            const float* out_tile_b = out.channel(2);
            
            float* dr = dstpR + yi * TILE_SIZE_Y * scale * dst_stride;
            float* dg = dstpG + yi * TILE_SIZE_Y * scale * dst_stride;
            float* db = dstpB + yi * TILE_SIZE_Y * scale * dst_stride;
            
            for (int y = 0; y < out.h; y++)
            {
                for (int x = 0; x < out.w; x++)
                {
                    dr[dst_stride * y + x] = std::min(1.f, std::max(0.f, out_tile_r[out.w * y + x] / 255.f));
                    dg[dst_stride * y + x] = std::min(1.f, std::max(0.f, out_tile_g[out.w * y + x] / 255.f));
                    db[dst_stride * y + x] = std::min(1.f, std::max(0.f, out_tile_b[out.w * y + x] / 255.f));
                }
            }
        }
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}
