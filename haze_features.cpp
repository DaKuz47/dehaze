#include <functional>

#include "haze_features.hpp"
#include "utils.h"
#include "math_features.hpp"
#include "atm_light.h"
#include "minimizer.h"

/* Приводит ядро к нужному виду для свёртки
*/
cv::Point prepareKernel(const cv::UMat& kernel_in, cv::UMat& kernel_out) {
	cv::flip(kernel_in, kernel_out, -1);
	cv::Point anchor(kernel_out.cols - kernel_out.cols / 2 - 1, kernel_out.rows - kernel_out.rows / 2 - 1);

	return anchor;
}


/* Функция, рассчитывает контрастную энергию для конкретного канала
@param image_clr входной канал
@param out_ce_clr выходная контрастная энергия
@param sigma средне квадратичное отклонение в фильтре гаусса
@param t пороговое значения для подавления шума
@param k коэффициент насыщения (semisaturatoin ?)
*/
void channelContrastEnergy(const cv::UMat& image_clr, cv::UMat& out_ce_clr, double sigma, double t, double k) {
	const int border_size = 20;
	cv::UMat bordered_img = make_border(image_clr, border_size);

	std::vector<double> kernel = gaussianKernel(sigma);
	cv::UMat y_kernel = cv::Mat(kernel).getUMat(cv::ACCESS_RW);
	cv::UMat x_kernel, conv_x, conv_y;
	cv::transpose(y_kernel, x_kernel);

	cv::UMat flipped_x_kernel, flipped_y_kernel;
	cv::Point anchorx = prepareKernel(x_kernel, flipped_x_kernel);
	cv::Point anchory = prepareKernel(y_kernel, flipped_y_kernel);
	cv::filter2D(bordered_img, conv_x, -1, flipped_x_kernel, anchorx, 0.0, cv::BORDER_CONSTANT);
	cv::filter2D(bordered_img, conv_y, -1, flipped_y_kernel, anchory, 0.0, cv::BORDER_CONSTANT);

	cv::add(conv_x.mul(conv_x), conv_y.mul(conv_y), bordered_img);
	cv::sqrt(bordered_img, bordered_img);
	bordered_img = remove_border(bordered_img, border_size);

	double z_max{};
    cv::minMaxIdx(bordered_img, nullptr, &z_max);

	cv::UMat temp;
	cv::add(bordered_img, z_max, temp);
	cv::multiply(temp, k, temp);
	cv::multiply(bordered_img, z_max, bordered_img);
	cv::divide(bordered_img, temp, bordered_img);
	cv::add(bordered_img, -t, bordered_img);
    cv::threshold(bordered_img, out_ce_clr, 0.0000001, 0, cv::THRESH_TOZERO);

}

// Функция, рассчитывает контрастную энергию для трёх цветовых каналов изображения
std::vector<cv::UMat> contrastEnergy(const cv::UMat& image_in) {
	assert(image_in.channels() == 3 && "not bgr image");

	const double sigma = 3.25; //средне квадратичное отклонение в фильтре гаусса
	const double k = 0.1;
	const double t_gray = 9.225496406318721e-4 * 255;
	const double t_by = 8.969246659629488e-4 * 255;
	const double t_rg = 2.069284034165411e-4 * 255;
	const int border_size = 20;

	cv::UMat image;
	image_in.convertTo(image, CV_64F);
	std::vector<cv::UMat> rgb_cnls;
	cv::split(image, rgb_cnls);

	cv::UMat temp1;
	cv::UMat temp2;
	cv::UMat temp3;
	cv::multiply(rgb_cnls[2], 0.299, temp1);
	cv::multiply(rgb_cnls[1], 0.587, temp2);
	cv::multiply(rgb_cnls[0], 0.114, temp3);
	cv::UMat gray_cnl;
	cv::add(temp1, temp2, gray_cnl);
	cv::add(gray_cnl, temp3, gray_cnl);

	cv::multiply(rgb_cnls[2], 0.5, temp1);
	cv::multiply(rgb_cnls[1], 0.5, temp2);
	cv::UMat by_cnl;
	cv::add(temp1, temp1, by_cnl);
	cv::subtract(by_cnl, rgb_cnls[0], by_cnl);
	cv::UMat rg_cnl;
	cv::subtract(rgb_cnls[2], rgb_cnls[1], rg_cnl);
	
	std::vector<cv::UMat> image_cnls = { gray_cnl, by_cnl, rg_cnl };
	std::vector<double> ce_t_coef = { t_gray, t_by, t_rg };
	for (int i = 0; i < image_cnls.size(); i++) {
		channelContrastEnergy(image_cnls[i], image_cnls[i], sigma, ce_t_coef[i], k);
	}

	return image_cnls;
}

// рассчитывает энтропию изображения, принимает серый канал изображения
double imageEntropy(const cv::UMat& gray, int patch_size) {
	assert(gray.channels() == 1 && "not gray image");

	cv::Mat hist;
	calc_cnl_hist(gray, hist);
	hist = hist / pow(patch_size, 2);
	
	double entropy = 0;
	for (int i = 0; i < hist.rows; i++) {
		double log_part = hist.at<float>(i, 0) == 0 ? 0 : log2(hist.at<float>(i, 0));
		entropy += hist.at<float>(i, 0) * log_part;
	}

	return -entropy;
}

// считает стандартное отклонение
double stdDeviation(const cv::UMat& gray, const cv::UMat& gaussKernel, double mu) {
	assert(gray.channels() == 1 && "not gray image");

	cv::UMat temp;
	cv::subtract(gray, mu, temp);
	cv::pow(temp, 2, temp);
	temp = temp.mul(gaussKernel);
	cv::sqrt(temp, temp);

	return cv::sum(temp)[0];
}

// считает нормализованную дисперсию
double normDisp(double std, double mu) {
	return std / (mu);
}

// Количественная оценка дымки в локальном участке изображения
double tmap(cv::Vec<double, 1> x, cv::UMat& patch, cv::Scalar a, int patch_size, cv::UMat& gaussianKernel) {
	double t = x[0];

	cv::UMat j;
	patch.convertTo(j, CV_64F);
	cv::subtract(j, a, j);
	cv::divide(j, t, j);
	cv::add(j, a, j);
	j.convertTo(j, CV_8U);
	cv::UMat jgray;
	cv::cvtColor(j, jgray, cv::COLOR_BGR2GRAY);

	cv::UMat square_j = j.reshape(3, patch_size);
	cv::transpose(square_j, square_j);
	std::vector<cv::UMat> contrast_energy = contrastEnergy(square_j);
	double entropy = imageEntropy(jgray, patch_size);
	jgray.convertTo(jgray, CV_64F, 1./255.);

	double mu = cv::sum(jgray.mul(gaussianKernel))[0];
	double stddev = stdDeviation(jgray, gaussianKernel, mu);
	double norm = normDisp(stddev, mu);

	std::vector<double> params = {
		entropy,
		cv::mean(contrast_energy[0])[0],
		cv::mean(contrast_energy[1])[0],
		cv::mean(contrast_energy[2])[0],
		stddev,
		norm
	};

	double res = 1;
	for (auto p : params) {
		res *= log(1 + abs(p));
	}

	res = res == 0.0 ? 0.00000001 : res;
	
	return -res;
}

// Поиск оптимальной карты пропускания
using namespace std::placeholders;
cv::Mat tmap_optimal(cv::UMat& patches, cv::Scalar a, int patch_size, int max_iter, double eps, bool log) {
	cv::Mat t_opt = cv::Mat::zeros(1, patches.cols, CV_64F);
	cv::Mat gauss = cv::getGaussianKernel(patch_size, patch_size / 4, CV_64F);
	cv::mulTransposed(gauss, gauss, false);
	std::vector<int> shape = { patch_size * patch_size, 1 };
	cv::Mat sheped_gauss = gauss.reshape(1, shape);
	cv::UMat fast_gaus = sheped_gauss.getUMat(cv::ACCESS_RW);

	shape = { patches.rows * patches.channels() };
	for (int patch = 0; patch < patches.cols; patch++) {
		cv::UMat i = patches(cv::Range(0, patches.rows), cv::Range(patch, patch + 1));
		double min_val{};
		cv::Mat tmp_patch;
		i.convertTo(tmp_patch, CV_64F);
		tmp_patch = tmp_patch / a;
		cv::minMaxIdx(tmp_patch.reshape(1, shape), &min_val);
		
		cv::Vec<double, 1> xmin(1-min_val);
		auto tmap_binded = bind(tmap, _1, i, a, patch_size, fast_gaus);

		auto start = std::clock();
		auto res = Nelder_Mead_Optimizer<decltype(tmap_binded), 1>(tmap_binded, xmin, 1);
		auto end = std::clock();

		if (log) {
			std::cout << patch << ": " << xmin[0] << ", Time (sec): " << (end-start)/1000.0 << "\n" << std::endl;
		}
		
		t_opt.at<double>(0, patch) = res[0];

	}

	return t_opt;
}
