#ifndef FILTER_H_
#define FILTER_H_

#include <opencv2/opencv.hpp>

class Filter {
 public:
  Filter(unsigned int window_size);
  bool Process(const cv::Mat& in, cv::Mat& out);

 private:
  bool processROI(const cv::Mat& in_roi, cv::Mat& out_roi);
  unsigned int window_size_;
};

#endif  // FILTER_H_
