#ifndef CudaUtils_H
#define CudaUtils_H

#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

// Copy data from h_particles to device and return d_particles
// d_particles will be allocated on device
// free with device_free_particles(d_particles)
particles* device_alloc_particles(particles* h_particles);

// Free a particles object allocated on device
void device_free_particles(particles* d_part);

// Copy memory from device to host for particles object
void copy_to_host_particles(particles* h_particles, particles* d_particles_);

// Copy data from h_field to device and return d_field
// d_field will be allocated on device
// free with device_free_field(d_field)
EMfield* device_alloc_EMfield(EMfield* h_field, long len);

// Free field allocated on device
void device_free_EMfield(EMfield* d_field) ;

// Copy field data back to host
void copy_to_host_EMfield(EMfield* h_field, EMfield* d_field_, long len);

// Copy data from h_grid to device and return d_grid
// d_grid will be allocated on device
// free with device_free_grid(d_grid)
grid* device_alloc_grid(grid* h_grid, long len);

// copy grid back to host
void copy_to_host_grid(grid* h_grid, grid* d_grid_, long len);

// free device grid memory
void device_free_grid(grid* d_grid);

// Copy data from h_param to device and return d_param
// d_param will be allocated on device
// free with device_free_param(d_param)
parameters* device_alloc_parameters(parameters* h_param);

// copy param back to host
void copy_to_host_parameters(parameters* h_param, parameters* d_param);

// free device param memory
void device_free_parameters(parameters* d_param);

#endif