all: optix_rt.ptx common.cuh vector.cuh matrix.cuh params.cuh mesh.cuh disk.cuh

optix_rt.ptx: optix_rt.cu common.cuh vector.cuh matrix.cuh params.cuh mesh.cuh disk.cuh
	nvcc optix_rt.cu -I /opt/optix-7.0.0/include/ -O3 -gencode arch=compute_61,code=compute_61 --ptx

clean:
	rm -f optix_rt.ptx
