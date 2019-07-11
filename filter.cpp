#include "chi2inv.h"
#include "filter.h"

#include <cmath>
#include <vector>

//#include <opencv2/core/utility.hpp>

template <typename T>
void printVector(std::vector<T>& v) {
    std::cout << "Vector: [ ";
    for (const auto& k: v) {
        std::cout << k << " ";
    }
    std::cout << " ]\n";
}

int roundUp(int value, unsigned int factor) {
  int temp = value + factor - 1;
  return temp - temp % factor;
}

//template <typename T1, typename T2>
//void calcNormalHist(std::vector<T1>& hist, const T2 mean, const T2 stddev,
//		const int min, const int max, const int bins, const int N) {
//	hist.resize(bins);
//	using FType = double;
//	FType left, right, left_value, right_value;
//	FType factor = 1/stddev/std::sqrt(2);
//	FType bin_size = FType(max-min)/bins;
//    FType offset = -0.5;  // Used to center intervals around integer values
//    auto normalized_erf = [&factor, &mean, &N](FType x) {
//        return 0.5 * N * std::erf((x-mean)*factor);
//	};

//    left = min + offset;
//    left_value = normalized_erf(left);
//	for (int i = 0; i < bins; ++i) {
//		right = left + bin_size;
//        right_value = normalized_erf(right);
//		hist[i] = right_value - left_value;
//		left = right;
//		left_value = right_value;
//	}
//}

template <typename T1, typename T2>
T1 calcPearsonsStatistic(const std::vector<T1>& probability, cv::Mat& hist) {
    // TODO: Check sizes
	T1 s = 0;
	const T1 eps = 0.001;
    auto&& it = hist.begin<T2>();
	for (const auto& p: probability) {
        if (p > eps) {
            s += std::pow(T1(*it) - p, 2)/p;
		}
		++it;
	}
    std::cout << "]\n";
	return s;
}

Filter::Filter(unsigned int window_size) : window_size_(window_size) {}

void testChi2Inv() {
    std::vector<double> alpha {0.95, 0.90, 0.80, 0.70, 0.50, 0.30,
                               0.20, 0.10, 0.05, 0.01, 0.001};
    for (int i = 1; i <= 10; ++i) {
        std::cout << "Degree: " << i << std::endl;
        for (const auto& a: alpha) {
            std::cout << chi2inv(1-a, i) << " ";
        }
        std::cout << std::endl;
    }
}

Filter::Filter(const unsigned int window_size, const FType alpha,
               const unsigned int bins): bins_(bins), window_size_(window_size) {
    // TODO: Check parameters
    // We need function values for degrees of freedom less than 'bins-2'
    chi2inv_table_.resize(bins);
    for (int d = 0; d < bins; ++d) {
        chi2inv_table_[d] = chi2inv(1-alpha, d);
    }
}

bool Filter::Process(const cv::Mat& input, cv::Mat& output) {
  testChi2Inv();
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


void Filter::calcNormalHist(std::vector<FType> &hist,
                            const FType mean, const FType stddev)
{
    hist.resize(bins);
    using FType = double;
    FType left, right, left_value, right_value;
    FType bin_size = FType(kHigh_- kLow_)/bins_;
    FType offset = -0.5;  // Used to center intervals around integer values
    FType factor = 1/stddev/std::sqrt(2);

    auto cumulative_frequency = [&factor, &mean, &N](FType x) {
        return 0.5 * window_size_ * window_size_ * std::erf((x-mean)*factor);
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

bool Filter::processROI(const cv::Mat& in_roi, cv::Mat& out_roi) {
  cv::Mat mean, stddev;
  cv::meanStdDev(in_roi, mean, stddev);
  std::cout << "Mean: " << mean.type() << mean.size() << mean << ", stddev: "
            << stddev.type () << stddev.size() << stddev << std::endl;

  std::vector<float> normal_hist;
  const int low = 0;
  const int high = 256;
  float range[] = {low, high};
  const float* histRange = { range };
  std::vector<cv::Mat> hist(in_roi.channels());
  bool uniform = true, accumulate = false;
  const auto N = window_size_ * window_size_;
  for (int channel = 0; channel < in_roi.channels(); ++channel) {
      cv::calcHist(&in_roi, 1, &channel, cv::Mat(),
                   hist[channel], 1, &bins_, &histRange,
                   uniform, accumulate);
      cv::Mat temp;
      cv::transpose(hist[channel], temp);
      std::cout << "Hist: " << hist[channel].size() << temp.type()
                << temp << std::endl;

      calcNormalHist(normal_hist, mean.at<double>(channel),
                     stddev.at<double>(channel), low, high, bins_, N);
      printVector(normal_hist);
      calcPearsonsStatistic<float, float>(normal_hist, hist[channel]);
  }


  std::cout << "ROI: " << in_roi.size() << in_roi << std::endl;
  cv::imshow("roi", in_roi);
  cv::waitKey(0);
  return true;
}


