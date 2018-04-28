#include "library.h"
#include <stdio.h>
#include <string.h>
#include <gsl/gsl_spline.h>


// Transforms a post-NMO CMP gather into the Radon Domain.
void radon_transform(radon_parameters_t *params, const double *data, double *rad_domain_out)
{
    printf("Beginning Radon transform...\n");
    // Intermediate values for the Radon Transform
    double t0, time, p, offset, amp;
    int num_time_steps = params->num_time_steps;
    double max_time = (num_time_steps - 1) * params->delta_t;
    int num_p = params->num_p;

    // Ensures rad_domain_out is zeroed
    memset(rad_domain_out, 0, sizeof(double) * num_p * num_time_steps);

    // Setup for Spline Interpolation
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, (size_t) num_time_steps);
    double *interp_temp_amp = malloc(sizeof(double) * ((size_t) num_time_steps));
    double *t_values = malloc(sizeof(double) * ((size_t) num_time_steps));

    for (int i = 0; i < num_time_steps; i++) {
        t_values[i] = params->delta_t * i;
    }

    for (int it = 0; it < num_time_steps; it++) {
        t0 = it * params->delta_t;

        for (int ir = 0; ir < params->num_receivers; ir++) {
            offset = params->min_offset + ir * params->delta_offset;

            // Initializing Spline for this column of amplitudes
            for (int i = 0; i < num_time_steps; i++) {
                interp_temp_amp[i] = data[INDEX(i, ir, params->num_receivers)];
            }
            gsl_spline_init(spline, t_values, interp_temp_amp, (size_t) num_time_steps);


            for (int ip = 0; ip < num_p; ip++) {
                p = params->p_min + params->delta_p * ip;
                time = t0 + p * offset * offset;

                if (0 < time && time < max_time) {
                    amp = gsl_spline_eval(spline, time, acc);
                } else {
                    amp = 0;
                }

                rad_domain_out[INDEX(it, ip, num_p)] += amp;
            }
        }
    }

    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
    free(interp_temp_amp);
    free(t_values);
    printf("Finished Radon transform\n");
}