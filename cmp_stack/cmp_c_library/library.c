#include "library.h"
#include <math.h>
#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

void radon_transform(radon_parameters_t *params, const float *data, float *rad_domain_out)
{
    // Intermediate values for the Radon Transform
    double t0, time, p, sum, offset, amp;
    int time_index;
    int num_p = (int) ((params->p_max - params->p_min) / params->delta_p);

    // Setup for Spline Interpolation
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, );

    for (int it = 0; it < params->num_time_steps; it++) {
        t0 = it * params->delta_t;

        for (int ip = 0; ip < num_p; ip++) {
            sum = 0;
            p = params->p_min + params->delta_p * ip;

            for (int ir = 0; ir < params->num_receivers; ir++) {
                offset = params->min_offset + ir * params->delta_offset;
                time = t0 + p * offset * offset;
                time_index = (int) (time / params->delta_t) + 1;

                if (time_index < params->num_time_steps && time_index >= 0) {
                    amp = data[INDEX(time_index, ir, params->num_receivers)];
                }
            }
        }
    }
}