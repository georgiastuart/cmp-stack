#include "library.h"
#include <stdio.h>
#include <string.h>
#include <gsl/gsl_spline.h>
#include <math.h>


// Transforms a post-NMO CMP gather into the Radon Domain.
// if mode == 0, does full transform between p_max and p_min
// if mode == 1, keeps primaries and discards multiples (keeps left of p_cutoff)
// if mode == 2, keeps multiples and discards primaries (keeps right of p_cutoff)
void radon_transform(radon_parameters_t *params, const double *data, double *rad_domain_out, int mode)
{
    // Intermediate values for the Radon Transform
    double t0, time, p, offset, amp;
    double p_min, p_max;
    int num_time_steps = params->num_time_steps;
    double max_time = (num_time_steps - 1) * params->delta_t;
    int num_p = params->num_p;

    switch (mode) {
        case 1:
            p_min = params->p_min;
            p_max = params->p_cutoff;
            break;
        case 2:
            p_min = params->p_cutoff;
            p_max = params->p_max;
            break;
        default:
            p_min = params->p_min;
            p_max = params->p_max;
    }

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

    for (int ir = 0; ir < params->num_receivers; ir++) {
        offset = params->min_offset + ir * params->delta_offset;

        // Initializing Spline for this column of amplitudes
        for (int i = 0; i < num_time_steps; i++) {
            interp_temp_amp[i] = data[INDEX(i, ir, params->num_receivers)];
        }
        gsl_spline_init(spline, t_values, interp_temp_amp, (size_t) num_time_steps);

        for (int it = 0; it < num_time_steps; it++) {
            t0 = it * params->delta_t;

            for (int ip = 0; ip < num_p; ip++) {
                p = p_min + params->delta_p * ip;
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
}

// Performs normal moveout according to a pre-determined velocity field
void normal_move_out(nmo_parameters_t *params, const double *data, double *nmo_data)
{
    double t_0, velocity, offset, t_nmo, amp;
    int num_time_steps = params->num_time_steps;
    double max_time = (num_time_steps - 1) * params->delta_t;
    int num_rec = params->num_receivers;

    // Setup for Spline Interpolation
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, (size_t) num_time_steps);
    double *interp_temp_amp = malloc(sizeof(double) * ((size_t) num_time_steps));
    double *t_values = malloc(sizeof(double) * ((size_t) num_time_steps));

    for (int i = 0; i < num_time_steps; i++) {
        t_values[i] = params->delta_t * i;
    }

    // Assumes no move out above the seafloor
    for (int it = 0; it < params->seafloor; it++) {
        for (int ir = 0; ir < num_rec; ir++) {
            nmo_data[INDEX(it, ir, num_rec)] = data[INDEX(it, ir, num_rec)];
        }
    }

    for (int ir = 0; ir < num_rec; ir++) {
        offset = params->min_offset + params->delta_offset * ir;

        // Initializing Spline for this column of amplitudes
        for (int i = 0; i < num_time_steps; i++) {
            interp_temp_amp[i] = data[INDEX(i, ir, num_rec)];
        }
        gsl_spline_init(spline, t_values, interp_temp_amp, (size_t) num_time_steps);

        for (int it = params->seafloor; it < num_time_steps; it++) {
            t_0 = t_values[it];
            velocity = params->vnmo_interp[it];
            t_nmo = sqrt(t_0 * t_0 + (offset * offset) / (velocity * velocity));

            if (t_nmo > max_time) {
                amp = 0;
            } else {

                amp = gsl_spline_eval(spline, t_nmo, acc);
            }

            nmo_data[INDEX(it, ir, num_rec)] = amp;
        }

        gsl_interp_accel_reset(acc);
    }

    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
    free(interp_temp_amp);
    free(t_values);
}