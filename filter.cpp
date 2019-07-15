#include "filter.h"
#include "chi2inv.h"

#include <opencv2/opencv.hpp>

#include <cmath>
#include <vector>

int roundUp(int value, unsigned int factor) {
  int temp = value + factor - 1;
  return temp - temp % factor;
}

void printChi2Inv() {
  std::vector<double> alpha{0.95, 0.90, 0.80, 0.70, 0.50, 0.30,
                            0.20, 0.10, 0.05, 0.01, 0.001};
  for (int i = 1; i <= 10; ++i) {
    std::cout << "Degree: " << i << std::endl;
    for (const auto& a : alpha) {
      std::cout << chi2inv(1 - a, i) << " ";
    }
    std::cout << std::endl;
  }
}

template <typename FType>
const cv::Scalar Filter<FType>::kSmoothColor = cv::Scalar(128, 128, 128);

template <typename FType>
Filter<FType>::Filter(const unsigned int window_size, const FType alpha,
                      const unsigned int bins, ThreadPool& pool)
    : bins_(bins), window_size_(window_size), thread_pool_(pool) {
  // TODO: Check parameters

  // We need function values for degrees of freedom less than 'bins-2'
  chi2inv_table_.resize(bins);
  for (int d = 0; d < bins; ++d) {
    chi2inv_table_[d] = chi2inv(1 - alpha, d);
  }
}

struct ROIDescriptor {
  ROIDescriptor(cv::Mat& i, cv::Mat& o) : in(i), out(o) {}
  cv::Mat in, out;
};

template <typename FType>
bool Filter<FType>::Process(const cv::Mat& input, cv::Mat& output) {
  int right = roundUp(input.cols, window_size_) - input.cols;
  int bottom = roundUp(input.rows, window_size_) - input.rows;

  cv::Mat input_extended;
  cv::copyMakeBorder(input, input_extended, 0, bottom, 0, right,
                     cv::BORDER_REFLECT_101);

  cv::Mat in_roi, out_roi;
  cv::Mat output_extended;
  input_extended.copyTo(output_extended);
  cv::Rect roi_rectangle{0, 0, int(window_size_), int(window_size_)};
  for (int i = 0; i <= (input_extended.rows - window_size_);
       i += window_size_) {
    roi_rectangle.y = i;

    for (int j = 0; j <= (input_extended.cols - window_size_);
         j += window_size_) {
      roi_rectangle.x = j;
      in_roi = input_extended(roi_rectangle);
      out_roi = output_extended(roi_rectangle);

      auto descriptor = std::make_shared<ROIDescriptor>(in_roi, out_roi);
      thread_pool_.RunTask([descriptor, this]() {
        if (!processROI(descriptor->in, descriptor->out)) {
          std::cout << "Failed to process roi\n";
        }
      });
    }
  }

  thread_pool_.Join();
  output = output_extended(cv::Rect{0, 0, input.cols, input.rows});
  return true;
}

template <typename FType>
void Filter<FType>::calcNormalHist(std::vector<FType>& hist, const FType mean,
                                   const FType stddev) {
  hist.resize(bins_);
  FType left, right, left_value, right_value;
  FType bin_size = FType(kHigh_ - kLow_) / bins_;
  FType offset = -0.5;  // Used to center intervals around integer values
  FType factor = 1 / stddev / std::sqrt(2);

  auto cumulative_frequency = [&factor, &mean, this](FType x) {
    return 0.5 * std::pow(this->window_size_, 2) *
           std::erf((x - mean) * factor);
  };

  left = kLow_ + offset;
  left_value = cumulative_frequency(left);
  for (int i = 0; i < bins_; ++i) {
    right = left + bin_size;
    right_value = cumulative_frequency(right);
    hist[i] = right_value - left_value;
    left = right;
    left_value = right_value;
  }
}

template <typename FType>
bool Filter<FType>::processROI(const cv::Mat& in_roi, cv::Mat& out_roi) {
  cv::Mat mean, stddev;
  cv::meanStdDev(in_roi, mean, stddev);

  float range[] = {kLow_, kHigh_};
  const int hist_size[] = {int(bins_)};
  const float* hist_range = {range};
  std::vector<cv::Mat> hist(in_roi.channels());
  bool uniform = true, accumulate = false;
  bool is_smooth = true;

  for (int channel = 0; channel < in_roi.channels(); ++channel) {
    cv::calcHist(&in_roi, 1, &channel, cv::Mat(), hist[channel], 1, hist_size,
                 &hist_range, uniform, accumulate);
    cv::Mat temp;
    cv::transpose(hist[channel], temp);

    is_smooth = is_smooth && testStatistic<float>(
                                 hist[channel], FType(mean.at<double>(channel)),
                                 FType(stddev.at<double>(channel)));
  }

  if (is_smooth) {
    out_roi = kSmoothColor;
  }

  return true;
}

template <typename FType>
template <typename T>
bool Filter<FType>::testStatistic(const cv::Mat& hist, const FType mean,
                                  const FType stddev) {
  std::vector<FType> normal_hist;
  calcNormalHist(normal_hist, mean, stddev);

  FType s = 0;
  const FType eps = 0.001;
  auto&& it = hist.begin<T>();
  int counter = 0;
  for (const auto& p : normal_hist) {
    if (p > eps) {
      s += std::pow(FType(*it) - p, 2) / p;
      ++counter;
    }
    ++it;
  }

  constexpr int degree_offset = 3;  // See task description
  if (counter >= degree_offset) {
    auto threshold = chi2inv_table_[counter - degree_offset];

#if defined(FILTER_DEBUG)
    std::cout << "Statistic: " << s << ", threshold: " << threshold
              << ", counter: " << counter << std::endl;
#endif

    return s < threshold;
  } else {
    std::cout << "Warning: not enough information for estimation. "
                 "Try another parameters combination.\n";
    return false;
  }
}

template class Filter<float>;
template class Filter<double>;
