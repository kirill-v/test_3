#ifndef FILTER_H_
#define FILTER_H_

#include <opencv2/opencv.hpp>

template <typename FType>
class Filter {
 public:
  Filter(const unsigned int window_size, const FType alpha,
         const unsigned int bins);
  bool Process(const cv::Mat& in, cv::Mat& out);

 private:
  void calcNormalHist(std::vector<FType>& hist, const FType mean,
                      const FType stddev);
  bool processROI(const cv::Mat& in_roi, cv::Mat& out_roi);
  template <typename T>
  bool testStatistic(const cv::Mat& hist, const FType mean, const FType stddev);

  // Brightness range. The upper boundary is exclusive.
  static constexpr int kLow_ = 0;
  static constexpr int kHigh_ = 256;
  static const cv::Scalar kSmoothColor;

  const unsigned int bins_;
  std::vector<FType> chi2inv_table_;
  const unsigned int window_size_;
};

#endif  // FILTER_H_
