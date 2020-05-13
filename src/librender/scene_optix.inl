#include "librender_ptx.h"
#include <iomanip>

#include <mitsuba/render/optix_structs.h>
#include <mitsuba/render/optix_api.h>

NAMESPACE_BEGIN(mitsuba)

#if !defined(NDEBUG)
# define MTS_OPTIX_DEBUG 1
#endif

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag
              << "]: " << message << "\n";
}

struct OptixState {
    OptixDeviceContext context;
    OptixPipeline pipeline = nullptr;
    OptixModule module = nullptr;
    OptixProgramGroup program_groups[5];
    OptixShaderBindingTable sbt = {};
    OptixTraversableHandle accel;
    void* accel_buffer;
    void* params;
};

template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) EmptySbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

using RayGenSbtRecord   = EmptySbtRecord;
using MissSbtRecord     = EmptySbtRecord;
using HitGroupSbtRecord = SbtRecord<HitGroupData>;

MTS_VARIANT void Scene<Float, Spectrum>::accel_init_gpu(const Properties &/*props*/) {
    optix_init();

    Log(Info, "Building scene in OptiX ..");
    m_accel = new OptixState();
    OptixState &s = *(OptixState *) m_accel;
    CUcontext cuCtx = 0;  // zero means take the current context
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
#if !defined(MTS_OPTIX_DEBUG)
    options.logCallbackLevel          = 1;
#else
    options.logCallbackLevel          = 3;
#endif
    rt_check(optixDeviceContextCreate(cuCtx, &options, &s.context));

    // Pipeline generation
    {
        OptixPipelineCompileOptions pipeline_compile_options = {};
        OptixModuleCompileOptions module_compile_options = {};

        module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#if !defined(MTS_OPTIX_DEBUG)
        module_compile_options.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#else
        module_compile_options.optLevel         = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        module_compile_options.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

        pipeline_compile_options.usesMotionBlur        = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pipeline_compile_options.numPayloadValues      = 3;
        pipeline_compile_options.numAttributeValues    = 3;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

#if !defined(MTS_OPTIX_DEBUG)
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#else
        pipeline_compile_options.exceptionFlags =
              OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW
            | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH
            | OPTIX_EXCEPTION_FLAG_USER
            | OPTIX_EXCEPTION_FLAG_DEBUG;
#endif

        rt_check_log(optixModuleCreateFromPTX(
            s.context,
            &module_compile_options,
            &pipeline_compile_options,
            (const char *)optix_rt_ptx,
            optix_rt_ptx_size,
            optix_log_buffer,
            &optix_log_buffer_size,
            &s.module
        ));

        OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

        // TODO figure out which shape to add to the pipeline

        OptixProgramGroupDesc prog_group_descs[5];
        memset(prog_group_descs, 0, sizeof(prog_group_descs));

        prog_group_descs[0].kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        prog_group_descs[0].raygen.module            = s.module;
        prog_group_descs[0].raygen.entryFunctionName = "__raygen__rg";

        prog_group_descs[1].kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        prog_group_descs[1].miss.module            = s.module;
        prog_group_descs[1].miss.entryFunctionName = "__miss__ms";

        prog_group_descs[2].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        prog_group_descs[2].hitgroup.moduleCH            = s.module;
        prog_group_descs[2].hitgroup.entryFunctionNameCH = "__closesthit__mesh";

        // Disk program group
        prog_group_descs[3].kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        prog_group_descs[3].hitgroup.moduleCH            = s.module;
        prog_group_descs[3].hitgroup.entryFunctionNameCH = "__closesthit__disk";
        prog_group_descs[3].hitgroup.moduleIS            = s.module;
        prog_group_descs[3].hitgroup.entryFunctionNameIS = "__intersection__disk";

#if !defined(MTS_OPTIX_DEBUG)
        const unsigned int num_program_groups = 4;
#else
        prog_group_descs[4].kind                         = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
        prog_group_descs[4].hitgroup.moduleCH            = s.module;
        prog_group_descs[4].hitgroup.entryFunctionNameCH = "__exception__err";
        const unsigned int num_program_groups = 5;
#endif

        rt_check_log(optixProgramGroupCreate(
            s.context,
            prog_group_descs,
            num_program_groups,
            &program_group_options,
            optix_log_buffer,
            &optix_log_buffer_size,
            s.program_groups
        ));

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth          = 1;
#if defined(MTS_OPTIX_DEBUG)
        pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
        pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
        pipeline_link_options.overrideUsesMotionBlur = false;
        rt_check_log(optixPipelineCreate(
            s.context,
            &pipeline_compile_options,
            &pipeline_link_options,
            s.program_groups,
            num_program_groups,
            optix_log_buffer,
            &optix_log_buffer_size,
            &s.pipeline
        ));
    } // End pipeline generation

    // Shader Binding Table generation and acceleration data structure building
    {
        uint32_t shapes_count = m_shapes.size();
        void* records = cuda_malloc(sizeof(RayGenSbtRecord) + sizeof(MissSbtRecord) + sizeof(HitGroupSbtRecord) * shapes_count);

        RayGenSbtRecord raygen_sbt;
        rt_check(optixSbtRecordPackHeader(s.program_groups[0], &raygen_sbt));
        void* raygen_record = records;
        cuda_memcpy_to_device(raygen_record, &raygen_sbt, sizeof(RayGenSbtRecord));

        MissSbtRecord miss_sbt;
        rt_check(optixSbtRecordPackHeader(s.program_groups[1], &miss_sbt));
        void* miss_record = (char*)records + sizeof(RayGenSbtRecord);
        cuda_memcpy_to_device(miss_record, &miss_sbt, sizeof(MissSbtRecord));

        // Allocate hitgroup records array
        void* hitgroup_records = (char*)records + sizeof(RayGenSbtRecord) + sizeof(MissSbtRecord);

        uint32_t shape_index = 0;
        std::vector<HitGroupSbtRecord> hg_sbts(shapes_count);

        for (size_t i = 0; i < 2; i++) {
            for (Shape* shape: m_shapes) {
                // TODO this is ugly!
                if (i == 0 && !shape->is_mesh())
                    continue;
                if (i == 1 && shape->is_mesh())
                    continue;

                shape->optix_geometry();
                // TODO
                size_t program_group_idx = (shape->is_mesh() ? 2 : 3);
                // Setup the hitgroup record and copy it to the hitgroup records array
                rt_check(optixSbtRecordPackHeader(s.program_groups[program_group_idx], &hg_sbts[shape_index]));
                // Compute optix geometry for this shape
                shape->optix_hit_group_data(hg_sbts[shape_index].data);

                ++shape_index;
            }
        }

        // Copy HitGroupRecords to the GPU
        cuda_memcpy_to_device(hitgroup_records, hg_sbts.data(), shapes_count * sizeof(HitGroupSbtRecord));

        s.sbt.raygenRecord                = (CUdeviceptr)raygen_record;
        s.sbt.missRecordBase              = (CUdeviceptr)miss_record;
        s.sbt.missRecordStrideInBytes     = sizeof(MissSbtRecord);
        s.sbt.missRecordCount             = 1;
        s.sbt.hitgroupRecordBase          = (CUdeviceptr)hitgroup_records;
        s.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        s.sbt.hitgroupRecordCount         = shapes_count;

        accel_parameters_changed_gpu();
    } // End shader binding table generation and acceleration data structure building

    // Allocate params pointer
    s.params = cuda_malloc(sizeof(Params));

    // This will trigger the scatter calls to upload geometry to the device
    cuda_eval();

    // TODO: check if we still want to do run a dummy launch
}

MTS_VARIANT void Scene<Float, Spectrum>::accel_parameters_changed_gpu() {
    OptixState &s = *(OptixState *) m_accel;

    if (m_shapes.empty())
        return;

    std::vector<Shape*> shape_meshes, shape_others;
    for (Shape* shape: m_shapes) {
        if (shape->is_mesh())
            shape_meshes.push_back(shape);
        else
            shape_others.push_back(shape);
    }

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.motionOptions.numKeys = 0;

    auto build_gas = [&s, &accel_options](const std::vector<Shape*> &shape_subset) {
        uint32_t shapes_count = shape_subset.size();

        if (shapes_count == 0)
            return OptixTraversableHandle(0);

        std::vector<OptixBuildInput> build_inputs(shapes_count);
        for (size_t i = 0; i < shapes_count; i++)
            shape_subset[i]->optix_build_input(build_inputs[i]);

        OptixAccelBufferSizes buffer_sizes;
        rt_check(optixAccelComputeMemoryUsage(s.context, &accel_options, build_inputs.data(), shapes_count, &buffer_sizes));

        void* d_temp_buffer   = cuda_malloc(buffer_sizes.tempSizeInBytes);
        void* d_output_buffer = cuda_malloc(buffer_sizes.outputSizeInBytes + 8);

        OptixAccelEmitDesc emit_property = {};
        emit_property.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_property.result = (CUdeviceptr)((char*)d_output_buffer + buffer_sizes.outputSizeInBytes);

        OptixTraversableHandle accel;
        rt_check(optixAccelBuild(
            s.context,
            0,              // CUDA stream
            &accel_options,
            build_inputs.data(),
            shapes_count,   // num build inputs
            (CUdeviceptr)d_temp_buffer,
            buffer_sizes.tempSizeInBytes,
            (CUdeviceptr)d_output_buffer,
            buffer_sizes.outputSizeInBytes,
            &accel,
            &emit_property,  // emitted property list
            1                // num emitted properties
        ));

        cuda_free((void*)d_temp_buffer);
        // cuda_free((void*)d_output_buffer); // TODO need to free this at some point

        return accel;

        // TODO: check if this is really usefull considering enoki's way of handling GPU memory
        // if (s.accel_buffer)
            // cuda_free((void*)s.accel_buffer);

        // size_t compacted_gas_size;
        // cuda_memcpy_from_device(&compacted_gas_size, (void*)emit_property.result, sizeof(size_t));
        // if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
        //     s.accel_buffer = cuda_malloc(compacted_gas_size);

        //     // Use handle as input and output
        //     rt_check(optixAccelCompact(s.context, 0, s.accel, (CUdeviceptr)s.accel_buffer, compacted_gas_size, &s.accel));

        //     cuda_free((void*)d_output_buffer);
        // } else {
        //     s.accel_buffer = d_output_buffer;
        // }
    };

    OptixTraversableHandle meshes_accel = build_gas(shape_meshes);
    OptixTraversableHandle others_accel = build_gas(shape_others);

    if (!shape_others.empty() && shape_meshes.empty()) {
        s.accel = others_accel;
    } else if (shape_others.empty() && !shape_meshes.empty()) {
        s.accel = meshes_accel;
    } else {
        // Create two empty instance with the identity transform
        OptixInstance instances[2] = {
            { {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, // transform
               0, 0, 255, OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM, 0, {0, 0}},
            { {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, // transform
               1, 0, 255, OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM, 0, {0, 0}}
        };

        instances[0].traversableHandle = meshes_accel;
        instances[0].sbtOffset = 0;
        instances[1].traversableHandle = others_accel;
        instances[1].sbtOffset = shape_meshes.size();

        void* d_instances = cuda_malloc(2 * sizeof(OptixInstance));
        cuda_memcpy_to_device(d_instances, &instances, 2 * sizeof(OptixInstance));

        OptixBuildInput build_input;
        build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        build_input.instanceArray.instances = d_instances;
        build_input.instanceArray.numInstances = 2;
        build_input.instanceArray.aabbs = 0;
        build_input.instanceArray.numAabbs = 0;

        OptixAccelBufferSizes buffer_sizes;
        rt_check(optixAccelComputeMemoryUsage(s.context, &accel_options, &build_input, 1, &buffer_sizes));

        void* d_temp_buffer   = cuda_malloc(buffer_sizes.tempSizeInBytes);
        void* d_output_buffer = cuda_malloc(buffer_sizes.outputSizeInBytes);

        rt_check(optixAccelBuild(
            s.context,
            0,              // CUDA stream
            &accel_options,
            &build_input,
            1,              // num build inputs
            (CUdeviceptr)d_temp_buffer,
            buffer_sizes.tempSizeInBytes,
            (CUdeviceptr)d_output_buffer,
            buffer_sizes.outputSizeInBytes,
            &s.accel,
            0,  // emitted property list
            0   // num emitted properties
        ));

        cuda_free((void*)d_temp_buffer);
        // cuda_free((void*)d_output_buffer); // TODO need to free this at some point
    }
}

MTS_VARIANT void Scene<Float, Spectrum>::accel_release_gpu() {
    OptixState &s = *(OptixState *) m_accel;
    cuda_free((void*)s.sbt.raygenRecord);
    cuda_free((void*)s.accel_buffer); // TODO
    cuda_free((void*)s.params);
    rt_check(optixPipelineDestroy(s.pipeline));
    rt_check(optixProgramGroupDestroy(s.program_groups[0]));
    rt_check(optixProgramGroupDestroy(s.program_groups[1]));
    rt_check(optixProgramGroupDestroy(s.program_groups[2]));
    rt_check(optixProgramGroupDestroy(s.program_groups[3]));
#if defined(MTS_OPTIX_DEBUG)
    rt_check(optixProgramGroupDestroy(s.program_groups[4]));
#endif
    rt_check(optixModuleDestroy(s.module));
    rt_check(optixDeviceContextDestroy(s.context));
    optix_shutdown();
    delete (OptixState *) m_accel;
    m_accel = nullptr;
}

MTS_VARIANT typename Scene<Float, Spectrum>::SurfaceInteraction3f
Scene<Float, Spectrum>::ray_intersect_gpu(const Ray3f &ray_, Mask active) const {
    if constexpr (is_cuda_array_v<Float>) {
        Assert(!m_shapes.empty());
        OptixState &s = *(OptixState *) m_accel;
        Ray3f ray(ray_);
        size_t ray_count = std::max(slices(ray.o), slices(ray.d));
        set_slices(ray, ray_count);
        set_slices(active, ray_count);

        SurfaceInteraction3f si = empty<SurfaceInteraction3f>(ray_count);

        // DEBUG mode: Explicitly instantiate `si` with NaN values.
        // As the integrator should only deal with the lanes of `si` for which
        // `si.is_valid()==true`, this makes it easier to catch bugs in the
        // masking logic implemented in the integrator.
#if !defined(NDEBUG)
            #define SET_NAN(name) name = full<decltype(name)>(std::numeric_limits<scalar_t<Float>>::quiet_NaN(), ray_count);
            SET_NAN(si.t); SET_NAN(si.time); SET_NAN(si.p); SET_NAN(si.uv); SET_NAN(si.n);
            SET_NAN(si.sh_frame.n); SET_NAN(si.dp_du); SET_NAN(si.dp_dv);
            #undef SET_NAN
#endif  // !defined(NDEBUG)

        cuda_eval();

        const Params params = {
            // Active mask
            active.data(),
            // In: ray origin
            ray.o.x().data(), ray.o.y().data(), ray.o.z().data(),
            // In: ray direction
            ray.d.x().data(), ray.d.y().data(), ray.d.z().data(),
            // In: ray extents
            ray.mint.data(), ray.maxt.data(),
            // Out: Distance along ray
            si.t.data(),
            // Out: UV coordinates
            si.uv.x().data(), si.uv.y().data(),
            // Out: Geometric normal
            si.n.x().data(), si.n.y().data(), si.n.z().data(),
            // Out: Shading normal
            si.sh_frame.n.x().data(), si.sh_frame.n.y().data(), si.sh_frame.n.z().data(),
            // Out: Intersection position
            si.p.x().data(), si.p.y().data(), si.p.z().data(),
            // Out: Texture space derivative (U)
            si.dp_du.x().data(), si.dp_du.y().data(), si.dp_du.z().data(),
            // Out: Texture space derivative (V)
            si.dp_dv.x().data(), si.dp_dv.y().data(), si.dp_dv.z().data(),
            // Out: Shape pointer (on host)
            (unsigned long long*)si.shape.data(),
            // Out: Primitive index
            si.prim_index.data(),
            // Out: Hit flag
            nullptr,
            // top_object
            s.accel
        };

        cuda_memcpy_to_device(s.params, &params, sizeof(Params));

        size_t width = 1, height = ray_count;
        while (!(height & 1) && width < height) {
            width <<= 1;
            height >>= 1;
        }

        OptixResult rt = optixLaunch(
            s.pipeline,
            0, // default cuda stream
            (CUdeviceptr)s.params,
            sizeof(Params),
            &s.sbt,
            width,
            height,
            1 // depth
        );
        if (rt == OPTIX_ERROR_HOST_OUT_OF_MEMORY) {
            cuda_malloc_trim();
            rt = optixLaunch(
                s.pipeline,
                0, // default cuda stream
                (CUdeviceptr)s.params,
                sizeof(Params),
                &s.sbt,
                width,
                height,
                1 // depth
            );
        }
        rt_check(rt);

        si.time = ray.time;
        si.wavelengths = ray.wavelengths;
        si.instance = nullptr;
        si.duv_dx = si.duv_dy = 0.f;

        // Gram-schmidt orthogonalization to compute local shading frame
        si.sh_frame.s = normalize(
            fnmadd(si.sh_frame.n, dot(si.sh_frame.n, si.dp_du), si.dp_du));
        si.sh_frame.t = cross(si.sh_frame.n, si.sh_frame.s);

        // Incident direction in local coordinates
        si.wi = select(si.is_valid(), si.to_local(-ray.d), -ray.d);

        return si;
    } else {
        ENOKI_MARK_USED(ray_);
        ENOKI_MARK_USED(active);
        Throw("ray_intersect_gpu() should only be called in GPU mode.");
    }
}

MTS_VARIANT typename Scene<Float, Spectrum>::Mask
Scene<Float, Spectrum>::ray_test_gpu(const Ray3f &ray_, Mask active) const {
    if constexpr (is_cuda_array_v<Float>) {
        OptixState &s = *(OptixState *) m_accel;
        Ray3f ray(ray_);
        size_t ray_count = std::max(slices(ray.o), slices(ray.d));
        Mask hit = empty<Mask>(ray_count);

        set_slices(ray, ray_count);
        set_slices(active, ray_count);

        cuda_eval();

        const Params params = {
            // Active mask
            active.data(),
            // In: ray origin
            ray.o.x().data(), ray.o.y().data(), ray.o.z().data(),
            // In: ray direction
            ray.d.x().data(), ray.d.y().data(), ray.d.z().data(),
            // In: ray extents
            ray.mint.data(), ray.maxt.data(),
            // Out: Distance along ray
            nullptr,
            // Out: UV coordinates
            nullptr, nullptr,
            // Out: Geometric normal
            nullptr, nullptr, nullptr,
            // Out: Shading normal
            nullptr, nullptr, nullptr,
            // Out: Intersection position
            nullptr, nullptr, nullptr,
            // Out: Texture space derivative (U)
            nullptr, nullptr, nullptr,
            // Out: Texture space derivative (V)
            nullptr, nullptr, nullptr,
            // Out: Shape pointer (on host)
            nullptr,
            // Out: Primitive index
            nullptr,
            // Out: Hit flag
            hit.data(),
            // top_object
            s.accel
        };

        cuda_memcpy_to_device(s.params, &params, sizeof(params));

        size_t width = 1, height = ray_count;
        while (!(height & 1) && width < height) {
            width <<= 1;
            height >>= 1;
        }

        OptixResult rt = optixLaunch(
            s.pipeline,
            0, // default cuda stream
            (CUdeviceptr)s.params,
            sizeof(Params),
            &s.sbt,
            width,
            height,
            1 // depth
        );
        if (rt == OPTIX_ERROR_HOST_OUT_OF_MEMORY) {
            cuda_malloc_trim();
            rt = optixLaunch(
                s.pipeline,
                0, // default cuda stream
                (CUdeviceptr)s.params,
                sizeof(Params),
                &s.sbt,
                width,
                height,
                1 // depth
            );
        }
        rt_check(rt);

        return hit;
    } else {
        ENOKI_MARK_USED(ray_);
        ENOKI_MARK_USED(active);
        Throw("ray_test_gpu() should only be called in GPU mode.");
    }
}

NAMESPACE_END(msiuba)
