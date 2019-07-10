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

template <typename T1, typename T2>
void calcNormalProbability(std::vector<T1>& hist, const T2 mean, const T2 stddev,
		const int min, const int max, const int bins) {
	hist.resize(bins);
	using FType = double;
	FType left, right, left_value, right_value;
	FType factor = 1/stddev/std::sqrt(2);
	FType bin_size = FType(max-min)/bins;
	FType offset = -0.5;  // Used to center intervals around integer values
	left = min + offset;
	left_value = 0.5 * std::erf((left-mean)*factor);
	for (int i = 0; i < bins; ++i) {
		right = left + bin_size;
		right_value = 0.5 * std::erf((right-mean)*factor);
		hist[i] = right_value - left_value;
		left = right;
		left_value = right_value;
	}
}

template <typename T1, typename T2>
T1 calcPearsonsStatistic(const std::vector<T1>& probability, cv::Mat& hist,
		const int N) {
	T1 s = 0;
	const T1 eps = 0.001;
    auto&& it = hist.begin<T2>();
	T1 frequency;
    std::cout << "Frequency: [";
	for (const auto& p: probability) {
		if (p > eps) {
			frequency = T1(*it)/N;
            std::cout << frequency << " ";
			s += (p - frequency)*(p - frequency)/p;
		}
		++it;
	}
    std::cout << "]\n";
	return s;
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

  const int bins = 128;
  std::vector<float> normal_hist;
//  int channels[] = {0, 1, 2};
  float range[] = {0, 256};
  const float* histRange = { range };
  std::vector<cv::Mat> hist(in_roi.channels());
  bool uniform = true, accumulate = false;
  for (int channel = 0; channel < in_roi.channels(); ++channel) {
      cv::calcHist(&in_roi, 1, &channel, cv::Mat(),
                   hist[channel], 1, &bins, &histRange,
                   uniform, accumulate);
      cv::Mat temp;
      cv::transpose(hist[channel], temp);
      std::cout << "Hist: " << hist[channel].size() << temp.type() << temp << std::endl;

      calcNormalProbability(normal_hist, mean.at<double>(channel),
    		  stddev.at<double>(channel),
    		  range[0], range[1], bins);
      printVector(normal_hist);
      calcPearsonsStatistic<float, float>(normal_hist, hist[channel], window_size_ * window_size_);
  }


  std::cout << "ROI: " << in_roi.size() << in_roi << std::endl;
  cv::imshow("roi", in_roi);
  cv::waitKey(0);
  return true;
}
