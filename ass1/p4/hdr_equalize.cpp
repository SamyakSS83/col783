#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <string>
#include <omp.h>
#include <filesystem>

using namespace cv;
using namespace std;

vector<double> linear_interpolate(const vector<double>& x_values,
                                  const vector<double>& y_values,
                                  const vector<double>& query_points) {
    vector<double> result(query_points.size(), 0.0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)query_points.size(); i++) {
        double x = query_points[i];
        if (x <= x_values.front()) {
            result[i] = y_values.front();
        }
        else if (x >= x_values.back()) {
            result[i] = y_values.back();
        }
        else {
            // Binary search to find interval
            auto it = upper_bound(x_values.begin(), x_values.end(), x);
            int j = max(0, (int)(it - x_values.begin()) - 1);
            double x1 = x_values[j], x2 = x_values[j + 1];
            double y1 = y_values[j], y2 = y_values[j + 1];
            if (x2 == x1) {
                result[i] = y1;
            } else {
                double fraction = (x - x1) / (x2 - x1);
                result[i] = y1 + fraction * (y2 - y1);
            }
        }
    }
    return result;
}

Mat equalize_histogram_hdr(const Mat& image, double min_output = 0, double max_output = 256) {
    CV_Assert(image.depth() == CV_32F || image.depth() == CV_64F);

    if (image.channels() > 1) {
        vector<Mat> channels;
        split(image, channels);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)channels.size(); i++) {
            channels[i] = equalize_histogram_hdr(channels[i], min_output, max_output);
        }

        Mat result;
        merge(channels, result);
        return result;
    }

    Mat flat = image.reshape(1, 1);
    flat.convertTo(flat, CV_64F);
    vector<double> pixels((double*)flat.datastart, (double*)flat.dataend);
    int total_pixels = pixels.size();

    vector<double> sorted_pixels = pixels;
    sort(sorted_pixels.begin(), sorted_pixels.end());

    vector<double> pixel_ranks(total_pixels);
    iota(pixel_ranks.begin(), pixel_ranks.end(), 1);
    for (auto& r : pixel_ranks) r /= total_pixels;

    double output_range = max_output - min_output;
    vector<double> mapped_values(total_pixels);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total_pixels; i++) {
        mapped_values[i] = min_output + output_range * pixel_ranks[i];
    }

    vector<double> unique_pixels, unique_mapped;
    unique_pixels.reserve(total_pixels);
    unique_mapped.reserve(total_pixels);

    unique_pixels.push_back(sorted_pixels[0]);
    unique_mapped.push_back(mapped_values[0]);
    for (int i = 1; i < total_pixels; i++) {
        if (sorted_pixels[i] != sorted_pixels[i - 1]) {
            unique_pixels.push_back(sorted_pixels[i]);
            unique_mapped.push_back(mapped_values[i]);
        }
    }

    vector<double> equalized_flat = linear_interpolate(unique_pixels, unique_mapped, pixels);

    Mat equalized(image.size(), CV_64F);
    memcpy(equalized.data, equalized_flat.data(), equalized_flat.size() * sizeof(double));
    return equalized;
}

static string ensure_results_dir(const string &dir = "results") {
    std::filesystem::create_directories(dir);
    return dir;
}

static void save_with_gamma(const Mat &src, const string &filepath) {
    Mat src_f; src.convertTo(src_f, CV_32F);
    double minVal = 0, maxVal = 0; minMaxLoc(src_f, &minVal, &maxVal);
    Mat norm = (maxVal - minVal > 1e-9) ? (src_f - (float)minVal) / (float)(maxVal - minVal)
                                         : Mat::zeros(src_f.size(), CV_32F);
    Mat gamma_corrected; pow(max(norm, 0), 1.0 / 2.2, gamma_corrected);
    Mat save8; gamma_corrected.convertTo(save8, CV_8U, 255.0);
    imwrite(filepath, save8);
}

static void save_image_pair(const Mat &original, const Mat &equalized, const string &prefix) {
    const string outdir = ensure_results_dir();
    save_with_gamma(original, outdir + "/" + prefix + "_original.jpg");
    save_with_gamma(equalized, outdir + "/" + prefix + "_equalized.jpg");
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: ./hdr_equalize <hdr_filename>\n";
        return -1;
    }

    string filename = argv[1];
    Mat hdr_image = imread(filename, IMREAD_ANYDEPTH | IMREAD_COLOR);
    if (hdr_image.empty()) {
        cerr << "Error: Could not load " << filename << "\n";
        return -1;
    }

    Mat hdr_rgb; cvtColor(hdr_image, hdr_rgb, COLOR_BGR2RGB);

    // Equalize to [0, 256)
    Mat equalized_hdr = equalize_histogram_hdr(hdr_rgb, 0, 256);

    // Save under results/
    save_image_pair(hdr_rgb, equalized_hdr, "hdr");
    cout << "Saved results to ./results (hdr_original.jpg, hdr_equalized.jpg)\n";
    return 0;
}
