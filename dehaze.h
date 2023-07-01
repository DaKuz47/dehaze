#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cassert>
#include "atm_light.h"
#include "utils.h"
#include "haze_features.hpp"
#include "filtering.hpp"

/* �������� �� �������� �����
* @param img ������� BGR �����������
* @param pathc_size ������ ����, �� ������� ����������� ����������� ��� ���������
* @param max_iter ������������ ���������� �������� ��� ��������� ������� ����
* @param eps �������� ���������� ��������� ������� ����
* @param lambda ����������� ��� ���������� ����������� ������������ �����
* @param tmin ����������� ����� �� �������� �����������
* @param dp dehaze power ����������� ��� �������� �����
* @param log ����� �������� ����������� ����������� ����� �����������
*/
cv::Mat dehaze(const cv::Mat& img, int path_size = 16, int max_iter = 10e5, double eps = 10e-7,
			   int lamda = 4, double tmin = 0.2, double dp = 0.7, bool log = false);