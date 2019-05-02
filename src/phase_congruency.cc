/**
 * @file   phase_congruency.cpp
 * @author Carlos H. Villa Pinto (chvillap@gmail.com), modified by Felix Graule (felix.graule@gmail.com)
 *
 * @copyright Copyright (c) 2016 Carlos Henrique Villa Pinto
 * @license GPL v2.0
 */

#include "phase_congruency.h"

// OpenCV
#include <opencv2/highgui/highgui.hpp>


phase_congruency::phase_congruency(
                 std::string            filename_prefix,
                 float                 *input_image,
                 log_gabor_filter_bank *filter_bank,
                 triple<size_t>         sizes,
                 bool                  *input_mask,
                 float                  noise_threshold,
                 float                  noise_std,
                 float                  sigmoid_gain,
                 float                  sigmoid_cutoff) {
    set_filename_prefix(filename_prefix);
    set_input_image(input_image);
    set_filter_bank(filter_bank);
    set_sizes(sizes);
    set_input_mask(input_mask);
    set_noise_threshold(noise_threshold);
    set_noise_std(noise_std);
    set_sigmoid_gain(sigmoid_gain);
    set_sigmoid_cutoff(sigmoid_cutoff);
}


phase_congruency::~phase_congruency() {}


std::ostream& phase_congruency::print(std::ostream &os) const {
    os << "{"
       << "m_filename_prefix: " << m_filename_prefix << ", "
       << "m_filter_bank: "     << m_filter_bank     << ", "
       << "m_sizes: "           << m_sizes           << ", "
       << "m_input_image: "     << m_input_image     << ", "
       << "m_input_mask: "      << m_input_mask      << ", "
       << "m_noise_threshold: " << m_noise_threshold << ", "
       << "m_noise_std: "       << m_noise_std       << ", "
       << "m_sigmoid_gain: "    << m_sigmoid_gain    << ", "
       << "m_sigmoid_cutoff: "  << m_sigmoid_cutoff  << ", "
       << "}";

    return os;
}


cv::Mat phase_congruency::compute_overall_pc_map() {
    /*
     * TODO:
     * Parallelize this method by dividing the computations below in multiple
     * threads. There are many places this can be done. For example, each
     * log-Gabor filter is applied to the input image independently. Moreover,
     * many computations are done pixelwise, so the pixel access could be
     * parallelized too. The performance could be greatly improved with that.
     */

    char filename[512];

    // This
    typedef itk::Vector<float, 3> TVector;

    // Get the total sizes in space and frequency domains.
    const size_t total_size   = m_sizes[0] * m_sizes[1] * m_sizes[2];
    const size_t f_total_size = total_size * sizeof(fftwf_complex);

    // Allocate data for the Fourier transform of the input image.
    fftwf_complex *f_input_image = (fftwf_complex*) fftwf_malloc(f_total_size);

    // Array of filtered images in the frequency domain.
    // It will contain one element per scale of the bank of filters.
    fftwf_complex *f_filtered_images = (fftwf_complex*)
        fftwf_malloc(f_total_size * m_filter_bank->get_num_scales());

    if (!f_input_image || !f_filtered_images)
        throw std::bad_alloc();

    // Compute the input image's Fourier transform (shifted).
    compute_shifted_FFT(f_input_image);

    // These arrays are used in the PC computation.
    float *sum_amplitude       = new float[total_size]();
    float *max_amplitude       = new float[total_size]();
    float *total_sum_amplitude = new float[total_size]();
    float *total_sum_energy    = new float[total_size]();

    #ifdef PHASE_CONGRUENCY_VERBOSE_ON
        std::cout << "Computing the PC maps...\n";
    #endif

    // Current orientation's index.
    size_t o = 0;

    // Get the step size in elevation angles.
    const float dtheta = (m_filter_bank->get_num_elevations() == 1) ?
                         0.0 :
                         M_PI_2 / (m_filter_bank->get_num_elevations() - 1);

    for (size_t e = 0; e < m_filter_bank->get_num_elevations(); ++e) {
        // Get the current elevation angle.
        const float theta     = e * dtheta;
        const float cos_theta = cos(theta);
        const float sin_theta = sin(theta);

        // Get the step size in azimuth angles.
        const float dphi = (m_filter_bank->get_num_azimuths() == 1) ?
                           0.0 :
                           (e == 0) ?
                               M_PI   / m_filter_bank->get_num_azimuths(0) :
                               M_PI*2 / m_filter_bank->get_num_azimuths(e);

        for (size_t a = 0; a < m_filter_bank->get_num_azimuths(e); ++a) {
            // Noise energy threshold.
            float T = 0.0;

            // Get the current azimuth angle.
            const float phi     = a * dphi;
            const float cos_phi = cos(phi);
            const float sin_phi = sin(phi);

            // The accumulated amplitudes are reset for each orientation.
            memset(sum_amplitude, 0, total_size * sizeof(float));
            memset(max_amplitude, 0, total_size * sizeof(float));

            for (size_t s = 0; s < m_filter_bank->get_num_scales(); ++s) {
                // Pointer to the filtered image at the current scale.
                fftwf_complex *p_f_filtered_image =
                    &f_filtered_images[s * total_size];

                // Apply a single log-Gabor filter (in the frequency domain) to
                // the input image. The result is stored in a slice of the
                // filtered images array.
                compute_filtering(p_f_filtered_image, f_input_image, s, a, e);

                // Accumulate amplitudes of filter responses over the scales.
                for (size_t i = 0; i < total_size; ++i) {
                    // Ignore locations outside the region of interest.
                    if (m_input_mask && !m_input_mask[i])
                        continue;

                    const float even      = p_f_filtered_image[i][0];
                    const float odd       = p_f_filtered_image[i][1];
                    const float amplitude = sqrt(sqr(even) + sqr(odd));

                    sum_amplitude[i] += amplitude;
                    max_amplitude[i] = std::max(amplitude, max_amplitude[i]);
                }

                // Automatic noise energy threshold estimation.
                if (m_noise_threshold < 0.0 && s == 0) {
                    T = estimate_noise_threshold(sum_amplitude);

                    #ifdef PHASE_CONGRUENCY_DEBUG_ON
                        std::cout << " (estimated T = " << T << ")";
                    #endif
                }

                #ifdef PHASE_CONGRUENCY_VERBOSE_ON
                    std::cout << " - done\n";
                #endif
            }

            for (size_t i = 0; i < total_size; ++i) {
                // Ignore locations outside the region of interest.
                if (m_input_mask && !m_input_mask[i])
                    continue;

                // Accumulate the even and odd filter responses over scales.
                float sum_even = 0.0;
                float sum_odd  = 0.0;
                for (size_t s = 0; s < m_filter_bank->get_num_scales(); ++s)
                {
                    // Pointer to the filtered image at the current scale.
                    fftwf_complex *p_f_filtered_image =
                        &f_filtered_images[s * total_size];

                    // Accumulate the even and odd filter responses.
                    sum_even += p_f_filtered_image[i][0];
                    sum_odd  += p_f_filtered_image[i][1];
                }

                // Get the mean filter responses over scales.
                float norm      = sqrt(sqr(sum_even) + sqr(sum_odd));
                float mean_even = sum_even / (norm + EPSILON);
                float mean_odd  = sum_odd  / (norm + EPSILON);

                // Compute the local energy response for the current
                // orientation.
                float local_energy = 0.0;
                for (size_t s = 0; s < m_filter_bank->get_num_scales(); ++s)
                {
                    // Pointer to the filtered image at the current scale.
                    fftwf_complex *p_f_filtered_image =
                        &f_filtered_images[s * total_size];

                    float even = p_f_filtered_image[i][0];
                    float odd  = p_f_filtered_image[i][1];

                    // Theoretically, we need to compute the product of the
                    // amplitude of the filter responses by the phase deviation
                    // at the current pixel.
                    // In practice, this is the same as computing:
                    local_energy += (even * mean_even + odd * mean_odd) -
                                fabs(even * mean_odd - odd * mean_even);
                }

                // Apply the noise energy threshold to the local energy
                // (either an automatically calculated threshold or a manually
                // given one).
                if (m_noise_threshold < 0.0)
                    local_energy -= T;
                else
                    local_energy -= m_noise_threshold;

                // Apply the local energy weighting.
                local_energy = apply_energy_weighting(
                    local_energy, sum_amplitude[i], max_amplitude[i]);

                // Accumulate the total sums in amplitude and energy along
                // all orientations.
                total_sum_amplitude[i] += sum_amplitude[i];
                total_sum_energy[i]    += local_energy;
            }
            ++o; // Move on to the next orientation.
        }
    }
    // Compute the final phase congruency map.
    cv::Mat pc_map(cv::Size(m_sizes[0], m_sizes[1]), CV_32F);
    for (size_t i = 0; i < total_size; ++i)
        pc_map.at<float>(i) = total_sum_energy[i] / (total_sum_amplitude[i] + EPSILON);

    // Write the final phase congruency map.
    sprintf(filename, "%s_PC.jpg", m_filename_prefix.c_str());
    pc_map.convertTo(pc_map, CV_8U, 255.0);

    // Clean up the memory.
    delete[] sum_amplitude;
    delete[] max_amplitude;
    delete[] total_sum_amplitude;
    delete[] total_sum_energy;
    fftwf_free(f_input_image);
    fftwf_free(f_filtered_images);
    fftwf_cleanup_threads();

    return pc_map;
}


std::vector<cv::Mat> phase_congruency::compute_eo_collection() {
    // Define collection data structure as vector of images
    std::vector<cv::Mat> eo_collection;

    // Iterate over all scales and all orientations
    for (int s = 0; s < m_filter_bank->get_num_scales(); s++){
        for (int o = 0; o < m_filter_bank->get_num_azimuths(); o++){

            // Get the total sizes in space and frequency domains.
            const size_t total_size   = m_sizes[0] * m_sizes[1];
            const size_t f_total_size = total_size * sizeof(fftwf_complex);

            // Allocate data for the Fourier transform of the input image.
            fftwf_complex *f_input_image = (fftwf_complex*) fftwf_malloc(f_total_size);

            // Allocate array of filtered image in the frequency domain.
            fftwf_complex *f_filtered_image = (fftwf_complex*) fftwf_malloc(f_total_size);

            if (!f_input_image || !f_filtered_image)
                throw std::bad_alloc();

            // Compute the input image's Fourier transform (shifted).
            compute_shifted_FFT(f_input_image);

            // Apply a single log-Gabor filter (in the frequency domain) to the input image.
            compute_filtering(f_filtered_image, f_input_image, s, o, 0);

            // Prepare OpenCV matrix for image to be stored in
            cv::Mat eo(cv::Size(m_sizes[0], m_sizes[1]), CV_32F);

            // Accumulate amplitudes of filter responses (real and imaginary part).
            for (size_t i = 0; i < total_size; ++i) {
                const float even = f_filtered_image[i][0];
                const float odd = f_filtered_image[i][1];

                const float absolute = std::sqrt(std::pow(even, 2) + std::pow(odd, 2));

                // Write pixel to overall image
                eo.at<float>(i) = absolute;
            }

            // Store full image in collection data structure
            eo_collection.push_back(eo);

            // Clean up allocated memory and threads from FFTW
            fftwf_free(f_filtered_image);
            fftwf_free(f_input_image);
            fftwf_cleanup_threads();
        }
    }
    return eo_collection;
}


void phase_congruency::compute_shifted_FFT(fftwf_complex *f_target) {
    /*
     * TODO:
     * Parallelize this method by dividing the computation of the nested loop
     * below in multiple threads, since each pixel is computed independently of
     * the others. The performance could be greatly improved with that.
     */

    #ifdef PHASE_CONGRUENCY_VERBOSE_ON
        std::cout << "Computing the FFT";
    #endif

    // Allow the use of multithreading in the FFT computation.
    fftwf_init_threads();
    fftwf_plan_with_nthreads(DEFAULT_FFTW_NUMBER_OF_THREADS);

    // Shift the DC component to the center of the image.
    // The target is initialized with the result.
    for (size_t i = 0, z = 0; z < m_sizes[2]; ++z)
        for (size_t y = 0; y < m_sizes[1]; ++y)
            for (size_t x = 0; x < m_sizes[0]; ++x, ++i) {
                f_target[i][0] = m_input_image[i] * pow(-1.0, x + y + z);
                f_target[i][1] = 0.0;
            }

    // Compute the forward FFT of the input image.
    FFT(f_target, m_sizes);

    // Save the resulting frequency spectrum.
    #ifdef PHASE_CONGRUENCY_DEBUG_ON
    {
        char filename[512];

        size_t total_size = m_sizes[0] * m_sizes[1] * m_sizes[2];
        float *f_spectrum = new float[total_size]();

        for (size_t i = 0; i < total_size; ++i)
            f_spectrum[i] = log(1 + sqrt(sqr(f_target[i][0]) +
                                         sqr(f_target[i][1])));

        normalize_min_max(f_spectrum, total_size, 0.0, 1.0);

        sprintf(filename, "%s_spectrum.nii", m_filename_prefix.c_str());
        write_image<float, 3>(filename,
            array2image<float, 3>(f_spectrum, m_sizes));

        delete[] f_spectrum;
    }
    #endif

    #ifdef PHASE_CONGRUENCY_VERBOSE_ON
        std::cout << " - done\n";
    #endif
}


void phase_congruency::compute_filtering(fftwf_complex *f_output,
                                         fftwf_complex *f_input,
                                         size_t         scale,
                                         size_t         azimuth,
                                         size_t         elevation) {
    /*
     * TODO:
     * Parallelize this method by dividing the computation of the nested loops
     * below in multiple threads, since each pixel is computed independently of
     * the others. The performance could be greatly improved with that.
     */

    // Get a single log-Gabor filter (in the frequency domain) for
    // the given scale, azimuth and elevation.
    float *f_filter = m_filter_bank->get_filter(scale, azimuth, elevation);

    #ifdef PHASE_CONGRUENCY_VERBOSE_ON
        std::cout << "   Processing filter: "
                  << "sc = "
                  << std::setfill('0') << std::setw(3) << scale
                  << ", az = "
                  << std::setfill('0') << std::setw(3) << azimuth
                  << ", el = "
                  << std::setfill('0') << std::setw(3) << elevation
                  << std::endl;
    #endif

    // Apply the log-Gabor filter.
    for (size_t i = 0, z = 0; z < m_sizes[2]; ++z)
        for (size_t y = 0; y < m_sizes[1]; ++y)
            for (size_t x = 0; x < m_sizes[0]; ++x, ++i) {
                f_output[i][0] = f_filter[i] * f_input[i][0];
                f_output[i][1] = f_filter[i] * f_input[i][1];
            }

    delete[] f_filter;

    // Compute the backward FFT in order to get the filtered image in the
    // space domain.
    FFT(f_output, m_sizes, true);

    // Shift the DC component back to its original location.
    for (size_t i = 0, z = 0; z < m_sizes[2]; ++z)
        for (size_t y = 0; y < m_sizes[1]; ++y)
            for (size_t x = 0; x < m_sizes[0]; ++x, ++i) {
                f_output[i][0] *= pow(-1.0, x + y + z);
                f_output[i][1] *= pow(-1.0, x + y + z);
            }

    // Save the filter responses (even, odd, amplitude).
    #ifdef PHASE_CONGRUENCY_DEBUG_ON
    {
        char filename[512];

        const size_t total_size = m_sizes[0] * m_sizes[1] * m_sizes[2];

        float *f_even      = new float[total_size]();
        float *f_odd       = new float[total_size]();
        float *f_amplitude = new float[total_size]();

        for (size_t i = 0; i < total_size; ++i) {
            f_even[i]      = f_output[i][0];
            f_odd[i]       = f_output[i][1];
            f_amplitude[i] = sqrt(sqr(f_even[i]) + sqr(f_odd[i]));
        }
        normalize_min_max(f_even,      total_size, -1.0, 1.0);
        normalize_min_max(f_odd,       total_size, -1.0, 1.0);
        normalize_min_max(f_amplitude, total_size,  0.0, 1.0);

        sprintf(filename, "%s_even_%02u_%02u_%02u.nii",
                m_filename_prefix.c_str(), scale, azimuth, elevation);
        write_image<float, 3>(filename,
            array2image<float, 3>(f_even, m_sizes));

        sprintf(filename, "%s_odd_%02u_%02u_%02u.nii",
                m_filename_prefix.c_str(), scale, azimuth, elevation);
        write_image<float, 3>(filename,
            array2image<float, 3>(f_odd, m_sizes));

        sprintf(filename, "%s_amplitude_%02u_%02u_%02u.nii",
                m_filename_prefix.c_str(), scale, azimuth, elevation);
        write_image<float, 3>(filename,
            array2image<float, 3>(f_amplitude, m_sizes));

        delete[] f_even;
        delete[] f_odd;
        delete[] f_amplitude;
    }
    #endif
}


float phase_congruency::estimate_noise_threshold(float *sum_amplitude) {
    const size_t total_size = m_sizes[0] * m_sizes[1] * m_sizes[2];

    float tau         = median(sum_amplitude, total_size) / sqrt(log(4.0));
    float invmult     = 1.0 / m_filter_bank->get_mult_factor();
    float nscales     = m_filter_bank->get_num_scales();
    float total_tau   = tau * (1.0 - pow(invmult, nscales)) / (1.0 - invmult);
    float noise_mean  = total_tau * sqrt(M_PI_2);
    float noise_sigma = total_tau * sqrt((4.0 - M_PI) / 2.0);

    return noise_mean + m_noise_std * noise_sigma;
}


float phase_congruency::apply_energy_weighting(float energy, float sum_amplitude, float max_amplitude) {
    if (energy > 0.0) {
        // Get the frequency range width.
        // If there is only one non-zero component, width is 0.
        // If all components are equal, width is 1.
        float width = (sum_amplitude /  (max_amplitude + EPSILON) - 1.0) /
                        (m_filter_bank->get_num_scales() - 1);

        // The weighting function is a sigmoid.
        float weight = 1.0 + exp(m_sigmoid_gain * (m_sigmoid_cutoff - width));
        energy /= weight;
    }
    else // Negative weights are simply set to 0.
        energy = 0.0;

    return energy;
}


// ----------------------------------------------------------------------------

std::ostream& operator<<(std::ostream &os, const phase_congruency &pc) {
    return pc.print(os);
}

