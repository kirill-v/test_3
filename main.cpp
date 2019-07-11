#include "filter.h"

#include <iostream>

#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
//  TODO: Add help

  cv::Mat input = cv::imread(argv[1]);
  if (input.empty()) {
    std::cerr << "Failed to load image from file: " << argv[1] << std::endl;
    return -1;
  }
  const auto window_size = static_cast<unsigned int>(std::stoul(argv[2]));
  const auto window_name = "Image";
  Filter filter{window_size};
  //    cv::namedWindow(window_name, cv::WINDOW_GUI_EXPANDED);
  cv::Mat output;
  filter.Process(input, output);
  //    cv::imshow(window_name, output);
  //    cv::waitKey(0);

  return 0;
}
