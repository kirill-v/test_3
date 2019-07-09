#include "filter.h"

//#include <opencv2/core/utility.hpp>

int roundUp(int value, unsigned int factor) {
  int temp = value + factor - 1;
  return temp - temp % factor;
}

Filter::Filter(unsigned int window_size) : window_size_(window_size) {}

bool Filter::Process(const cv::Mat& input, cv::Mat& output) {
  int right = roundUp(input.cols, window_size_) - input.cols;
  int bottom = roundUp(input.rows, window_size_) - input.rows;
  std::cout << "Right: " << right << ", bottom: " << bottom << std::endl;
  cv::Mat input_extended{};
  cv::copyMakeBorder(input, input_extended, 0, bottom, 0, right,
                     cv::BORDER_REFLECT_101);
  std::cout << "Extended: " << input_extended.size << std::endl;
  //  cv::imshow("Filter::Process", input_extended);
  //  cv::waitKey(0);

  cv::Mat roi, out_roi;
  auto output_type = CV_8U;
  cv::Mat output_extended(input_extended.size(),
                     CV_MAKETYPE(output_type, input.channels()));
  cv::Rect roi_rectangle{0, 0, int(window_size_), int(window_size_)};
  for (int i = 0; i <= (input_extended.rows - window_size_);
       i += window_size_) {
    roi_rectangle.y = i;
    for (int j = 0; j <= (input_extended.cols - window_size_);
         j += window_size_) {
      roi_rectangle.x = j;
      roi = input_extended(roi_rectangle);
      out_roi = output_extended(roi_rectangle);
      if (!processROI(roi, out_roi)) {
        std::cout << "Failed to process roi\n";
        return false;
      }
      //          roi.adjustROI(0, 0, -window_size_, window_size_);
      //          out_roi.adjustROI(0, 0, -window_size_, window_size_);
    }
  }
  output = input_extended;
  return true;
}

bool Filter::processROI(const cv::Mat& in_roi, cv::Mat& out_roi) {
//  auto mean = cv::mean(in_roi);
  cv::Mat mean, stddev;
  cv::meanStdDev(in_roi, mean, stddev);
  std::cout << "Mean: " << mean.size() << mean << ", stddev: " << stddev.size()
		  << stddev << std::endl;
//  cv::imshow("roi", in_roi);
//  cv::waitKey(0);
  return true;
}
