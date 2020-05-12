#pragma once

#include <math.h>
#include "common.cuh"
#include "params.cuh"

/* Compute and store information describing the intersection. This function
   is very similar to Disk::fill_surface_interaction() */
__device__ void compute_surface_interaction_disk(const HitGroupData *sbt_data,
                                                 Vector3f &p, Vector2f &uv,
                                                 Vector3f &ns, Vector3f &ng,
                                                 Vector3f &dp_du,
                                                 Vector3f &dp_dv,
                                                 float &t) {
    // Ray in world-space
    Vector3f ray_o_ = make_vector3f(optixGetWorldRayOrigin());
    Vector3f ray_d_ = make_vector3f(optixGetWorldRayDirection());

    Transform4f to_world(sbt_data->to_world);
    Transform4f to_object(sbt_data->to_object);

    // Ray in object-space
    Vector3f ray_o = transform_point(to_object, ray_o_);
    Vector3f ray_d = transform_vector(to_object, ray_d_);

    t = -ray_o.z() / ray_d.z();

    Vector3f local = ray_o + ray_d * t;

    float r = norm(Vector2f(local.x(), local.y())),
          inv_r = 1.f / r;

    float v = atan2(local.y(), local.x()) / (2.f * M_PI);
    if (v < 0.f)
        v += 1.f;

    float cos_phi = (r != 0.f ? local.x() * inv_r : 1.f),
          sin_phi = (r != 0.f ? local.y() * inv_r : 0.f);

    dp_du = transform_vector(to_world, Vector3f( cos_phi, sin_phi, 0.f));
    dp_dv = transform_vector(to_world, Vector3f(-sin_phi, cos_phi, 0.f));

    ns = normalize(transform_normal(to_world, Vector3f(0.f, 0.f, 1.f)));
    ng = ns;
    uv = Vector2f(r, v);
    p = ray_o_ + ray_d_ * t;
}


extern "C" __global__ void __intersection__disk() {
    const HitGroupData *sbt_data = (HitGroupData*) optixGetSbtDataPointer();

    unsigned int launch_index = calculate_launch_index();

    Transform4f to_world(sbt_data->to_world);
    Transform4f to_object(sbt_data->to_object);

    // if (launch_index == 686263) {
    //     printf("to_world.matrix: \n");
    //     for (size_t i = 0; i < 4; i++)
    //         printf("%f, %f, %f, %f \n",
    //                to_world.matrix[i][0],
    //                to_world.matrix[i][1],
    //                to_world.matrix[i][2],
    //                to_world.matrix[i][3]);

    //     printf("to_world.inverse_transpose: \n");
    //     for (size_t i = 0; i < 4; i++)
    //         printf("%f, %f, %f, %f \n",
    //                to_world.inverse_transpose[i][0],
    //                to_world.inverse_transpose[i][1],
    //                to_world.inverse_transpose[i][2],
    //                to_world.inverse_transpose[i][3]);

    //     printf("to_object: \n");
    //     for (size_t i = 0; i < 4; i++)
    //         printf("%f, %f, %f, %f \n",
    //                to_object.matrix[i][0],
    //                to_object.matrix[i][1],
    //                to_object.matrix[i][2],
    //                to_object.matrix[i][3]);

    //     printf("to_object.inverse_transpose: \n");
    //     for (size_t i = 0; i < 4; i++)
    //         printf("%f, %f, %f, %f \n",
    //                to_object.inverse_transpose[i][0],
    //                to_object.inverse_transpose[i][1],
    //                to_object.inverse_transpose[i][2],
    //                to_object.inverse_transpose[i][3]);
    // }

    Vector3f ray_o = make_vector3f(optixGetWorldRayOrigin());
    Vector3f ray_d = make_vector3f(optixGetWorldRayDirection());

    ray_o = transform_point(to_object, ray_o);
    ray_d = transform_vector(to_object, ray_d);

    float t = -ray_o.z() / ray_d.z();
    Vector3f local = ray_o + ray_d * t;

    if (local.x() * local.x() + local.y() * local.y() <= 1.f)
        optixReportIntersection(t, OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE);
}


extern "C" __global__ void __closesthit__disk() {
    unsigned int launch_index = calculate_launch_index();

    if (params.out_hit != nullptr) {
        params.out_hit[launch_index] = true;
    } else {
        Vector3f p;
        Vector2f uv;
        Vector3f ns;
        Vector3f ng;
        Vector3f dp_du;
        Vector3f dp_dv;
        float t;
        const HitGroupData *sbt_data = (HitGroupData *) optixGetSbtDataPointer();

        compute_surface_interaction_disk(sbt_data, p, uv, ns, ng, dp_du, dp_dv, t);

        write_output_params(params, launch_index,
                            sbt_data->shape_ptr,
                            optixGetPrimitiveIndex(),
                            p, uv, ns, ng, dp_du, dp_dv, t);
    }
}
