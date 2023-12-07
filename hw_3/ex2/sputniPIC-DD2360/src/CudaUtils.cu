#include "Particles.h"
#include "Alloc.h"
#include "CudaUtils.h"
#include <cuda.h>
#include <cuda_runtime.h>

particles* device_alloc_particles(particles* h_particles) 
{
    particles* d_particles = (particles*)malloc(sizeof(particles));
    memcpy(d_particles, h_particles, sizeof(particles));
    particles* ret_val;

    long npmax = d_particles->npmax;

    cudaMalloc(&d_particles->x, npmax * sizeof(FPpart));
    cudaMalloc(&d_particles->y, npmax * sizeof(FPpart));
    cudaMalloc(&d_particles->z, npmax * sizeof(FPpart));
    cudaMalloc(&d_particles->u, npmax * sizeof(FPpart));
    cudaMalloc(&d_particles->v, npmax * sizeof(FPpart));
    cudaMalloc(&d_particles->w, npmax * sizeof(FPpart));
    cudaMalloc(&ret_val, sizeof(particles));

    cudaMemcpy(d_particles->x, h_particles->x, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles->y, h_particles->y, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles->z, h_particles->z, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles->u, h_particles->u, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles->v, h_particles->v, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles->w, h_particles->w, npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(ret_val, d_particles, sizeof(particles), cudaMemcpyHostToDevice);

    free(d_particles);
    return ret_val;
}

void device_free_particles(particles* d_part)
{
    particles* h_part = (particles*)malloc(sizeof(particles));
    cudaMemcpy(h_part, d_part, sizeof(particles), cudaMemcpyDeviceToHost);

    cudaFree(h_part->x);
    cudaFree(h_part->y);
    cudaFree(h_part->z);
    cudaFree(h_part->u);
    cudaFree(h_part->v);
    cudaFree(h_part->w);

    cudaFree(d_part);
    free(h_part);
}

void copy_to_host_particles(particles* h_particles, particles* d_particles_)
{
    particles* d_particles = (particles*)malloc(sizeof(particles));
    cudaMemcpy(d_particles, d_particles_, sizeof(particles), cudaMemcpyDeviceToHost);

    long npmax = h_particles->npmax;
    cudaMemcpy(h_particles->x, d_particles->x, npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_particles->y, d_particles->y, npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_particles->z, d_particles->z, npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_particles->u, d_particles->u, npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_particles->v, d_particles->v, npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_particles->w, d_particles->w, npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);

    free(d_particles);
}

EMfield* device_alloc_EMfield(EMfield* h_field, long len)
{
    EMfield* d_field = (EMfield*)malloc(sizeof(EMfield));
    memcpy(d_field, h_field, sizeof(EMfield));
    EMfield* ret_val;

    cudaMalloc(&d_field->Ex_flat, len * sizeof(FPfield));
    cudaMalloc(&d_field->Ey_flat, len * sizeof(FPfield));
    cudaMalloc(&d_field->Ez_flat, len * sizeof(FPfield));
    cudaMalloc(&d_field->Bxn_flat, len * sizeof(FPfield));
    cudaMalloc(&d_field->Byn_flat, len * sizeof(FPfield));
    cudaMalloc(&d_field->Bzn_flat, len * sizeof(FPfield));
    cudaMalloc(&ret_val, sizeof(EMfield));

    cudaMemcpy(d_field->Ex_flat, h_field->Ex_flat, len * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_field->Ey_flat, h_field->Ey_flat, len * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_field->Ez_flat, h_field->Ez_flat, len * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_field->Bxn_flat, h_field->Bxn_flat, len * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_field->Byn_flat, h_field->Byn_flat, len * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_field->Bzn_flat, h_field->Bzn_flat, len * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(ret_val, d_field, sizeof(EMfield), cudaMemcpyHostToDevice);

    free(d_field);
    return ret_val;
}

void device_free_EMfield(EMfield* d_field) 
{
    EMfield* h_field = (EMfield*)malloc(sizeof(EMfield));
    cudaMemcpy(h_field, d_field, sizeof(EMfield), cudaMemcpyDeviceToHost);

    cudaFree(h_field->Ex_flat);
    cudaFree(h_field->Ey_flat);
    cudaFree(h_field->Ez_flat);
    cudaFree(h_field->Bxn_flat);
    cudaFree(h_field->Byn_flat);
    cudaFree(h_field->Bzn_flat);

    cudaFree(d_field);
    free(h_field);
}

void copy_to_host_EMfield(EMfield* h_field, EMfield* d_field_, long len)
{
    EMfield* d_field = (EMfield*)malloc(sizeof(EMfield));
    cudaMemcpy(d_field, d_field_, sizeof(EMfield), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_field->Ex_flat, d_field->Ex_flat, len * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_field->Ey_flat, d_field->Ey_flat, len * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_field->Ez_flat, d_field->Ez_flat, len * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_field->Bxn_flat, d_field->Bxn_flat, len * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_field->Byn_flat, d_field->Byn_flat, len * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_field->Bzn_flat, d_field->Bzn_flat, len * sizeof(FPfield), cudaMemcpyDeviceToHost);

    free(d_field);
}   

grid* device_alloc_grid(grid* h_grid, long len)
{
    grid* d_grid = (grid*)malloc(sizeof(grid));
    memcpy(d_grid, h_grid, sizeof(grid));
    grid* ret_val;

    cudaMalloc(&d_grid->XN_flat, len * sizeof(FPfield));
    cudaMalloc(&d_grid->YN_flat, len * sizeof(FPfield));
    cudaMalloc(&d_grid->ZN_flat, len * sizeof(FPfield));
    cudaMalloc(&ret_val, sizeof(grid));

    cudaMemcpy(d_grid->XN_flat, h_grid->XN_flat, len * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid->YN_flat, h_grid->YN_flat, len * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid->ZN_flat, h_grid->ZN_flat, len * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(ret_val, d_grid, sizeof(grid), cudaMemcpyHostToDevice);
    
    free(d_grid);
    return ret_val;
}

void copy_to_host_grid(grid* h_grid, grid* d_grid_, long len) 
{
    grid* d_grid = (grid*)malloc(sizeof(grid));
    cudaMemcpy(d_grid, d_grid_, sizeof(grid), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_grid->XN_flat, d_grid->XN_flat, len * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grid->YN_flat, d_grid->YN_flat, len * sizeof(FPfield), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grid->ZN_flat, d_grid->ZN_flat, len * sizeof(FPfield), cudaMemcpyDeviceToHost);

    free(d_grid);
}

void device_free_grid(grid* d_grid)
{
    grid* h_grid = (grid*)malloc(sizeof(grid));
    cudaMemcpy(h_grid, d_grid, sizeof(grid), cudaMemcpyDeviceToHost);

    cudaFree(h_grid->XN_flat);
    cudaFree(h_grid->YN_flat);
    cudaFree(h_grid->ZN_flat);

    cudaFree(d_grid);
    free(h_grid);
}

parameters* device_alloc_parameters(parameters* h_param)
{
    parameters* d_param;
    cudaMalloc(&d_param, sizeof(parameters));
    cudaMemcpy(d_param, h_param, sizeof(parameters), cudaMemcpyHostToDevice);
    return d_param;
}

void copy_to_host_parameters(parameters* h_param, parameters* d_param)
{
    cudaMemcpy(h_param, d_param, sizeof(parameters), cudaMemcpyDeviceToHost);
}

void device_free_parameters(parameters* d_param)
{
    cudaFree(d_param);
}
