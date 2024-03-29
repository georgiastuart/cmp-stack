#ifndef CMP_C_LIBRARY_LIBRARY_H
#define CMP_C_LIBRARY_LIBRARY_H

#define INDEX(y, x, dim) ((y) * (dim) + (x))

typedef struct radon_parameters
{
    int num_time_steps, num_receivers;
    double delta_t, delta_offset, min_offset;
    double p_min, p_max, p_cutoff, delta_p;
    int num_p;

} radon_parameters_t;

typedef struct nmo_parameters
{
    int num_time_steps, num_receivers;
    double delta_t, delta_offset, min_offset;
    int seafloor;
    double *vnmo_interp;
} nmo_parameters_t;

void radon_transform(radon_parameters_t *params, const double *data, double *rad_domain_out, int mode);
void normal_move_out(nmo_parameters_t *params, const double *data, double *nmo_data);

#endif