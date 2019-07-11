#ifndef FILTER_H_
#define FILTER_H_

#include <opencv2/opencv.hpp>

class Filter {
 public:
  using FType = double;
  Filter(const unsigned int window_size, const FType alpha,
         const unsigned int bins);
  bool Process(const cv::Mat& in, cv::Mat& out);

 private:
  void calcNormalHist(std::vector<FType>& hist, const FType mean, const FType stddev);
  bool processROI(const cv::Mat& in_roi, cv::Mat& out_roi);

  // Brightness range. The upper boundary is exclusive.
  static constexpr int kLow_ = 0;
  static constexpr int kHigh_ = 256;

  constexpr unsigned int bins_;
  std::vector<FType> chi2inv_table_;
  constexpr unsigned int window_size_;
};

#endif  // FILTER_H_
