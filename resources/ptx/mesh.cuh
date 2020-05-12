#pragma once

#include "common.cuh"
#include "params.cuh"
#include <optix.h>

/* Compute and store information describing the intersection. This function
   is very similar to Mesh::fill_surface_interaction() */
__device__ void compute_surface_interaction_mesh(const HitGroupData *sbt_data,
                                                 Vector3f &p, Vector2f &uv,
                                                 Vector3f &ns, Vector3f &ng,
                                                 Vector3f &dp_du,
                                                 Vector3f &dp_dv,
                                                 float &t) {
    float2 float2_uv = optixGetTriangleBarycentrics();
    uv  = Vector2f(float2_uv.x, float2_uv.y);
    float uv0 = 1.f - uv.x() - uv.y(),
          uv1 = uv.x(),
          uv2 = uv.y();

    const Vector3u *faces            = sbt_data->faces;
    const Vector3f *vertex_positions = sbt_data->vertex_positions;
    const Vector3f *vertex_normals   = sbt_data->vertex_normals;
    const Vector2f *vertex_texcoords = sbt_data->vertex_texcoords;

    Vector3u face = faces[optixGetPrimitiveIndex()];

    Vector3f p0 = vertex_positions[face.x()],
             p1 = vertex_positions[face.y()],
             p2 = vertex_positions[face.z()];

    Vector3f dp0 = p1 - p0,
             dp1 = p2 - p0;

    p = p0 * uv0 + p1 * uv1 + p2 * uv2;

    ng = normalize(cross(dp0, dp1));
    coordinate_system(ng, dp_du, dp_dv);

    if (vertex_normals != nullptr) {
        Vector3f n0 = vertex_normals[face.x()],
                 n1 = vertex_normals[face.y()],
                 n2 = vertex_normals[face.z()];

        ns = normalize(n0 * uv0 + n1 * uv1 + n2 * uv2);
    } else {
        ns = ng;
    }

    if (vertex_texcoords != nullptr) {
        Vector2f t0 = vertex_texcoords[face.x()],
                 t1 = vertex_texcoords[face.y()],
                 t2 = vertex_texcoords[face.z()];

        uv = t0 * uv0 + t1 * uv1 + t2 * uv2;

        Vector2f dt0 = t1 - t0,
                 dt1 = t2 - t0;
        float det = dt0.x() * dt1.y() - dt0.y() * dt1.x();

        if (det != 0.f) {
            float inv_det = 1.f / det;
            dp_du = ( dt1.y() * dp0 - dt0.y() * dp1) * inv_det;
            dp_dv = (-dt1.x() * dp0 + dt0.x() * dp1) * inv_det;
        }
    }

    Vector3f ray_o = make_vector3f(optixGetWorldRayOrigin());
    Vector3f ray_d = make_vector3f(optixGetWorldRayDirection());
    t = sqrt(squared_norm(p - ray_o) / squared_norm(ray_d));
}

extern "C" __global__ void __closesthit__mesh() {
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

        compute_surface_interaction_mesh(sbt_data, p, uv, ns, ng, dp_du, dp_dv, t);

        write_output_params(params, launch_index,
                            sbt_data->shape_ptr,
                            optixGetPrimitiveIndex(),
                            p, uv, ns, ng, dp_du, dp_dv, t);
    }
}
