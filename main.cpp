#include "filter.h"
#include "thread_pool.h"

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
  const cv::String keys =
      "{help h ? |      | print this message                         }"
      "{@image   |<none>| input image                                }"
      "{size     |  11  | size of processing window                  }"
      "{alpha    | 0.05 | significance level from interval(0,1). "
      "Try pass (1-alpha) if result looks strange.                   }"
      "{bins     |  64  | number of bins in a histogram: 32, 64, 128 }";

  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("filter-test");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  std::string input_name = parser.get<cv::String>("@image");
  if (input_name.empty()) {
    std::cerr << "Error: You should specify input image path\n";
    parser.printMessage();
    return -1;
  }
  const auto window_size = parser.get<unsigned int>("size");
  const auto alpha = parser.get<double>("alpha");
  const auto bins = parser.get<unsigned int>("bins");

  if (!parser.check()) {
    parser.printErrors();
    parser.printMessage();
    return -1;
  }

  cv::Mat input = cv::imread(input_name, cv::IMREAD_UNCHANGED);
  if (input.empty()) {
    std::cerr << "Failed to load image from file: " << input_name << "\n";
    return -1;
  }

  ThreadPool pool{std::thread::hardware_concurrency()};
  Filter<float> filter(window_size, alpha, bins, pool);
  cv::Mat output;
  filter.Process(input, output);

  cv::imshow("input", input);
  cv::imshow("output", output);
  cv::waitKey(0);

  return 0;
}
