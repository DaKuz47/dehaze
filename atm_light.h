#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

/* ��������� ����� ������������ ����� ������� QuadTree division. BGR �����������
* @param img ������� �����������
* @param stopdiv_size ������������� ������ ��������� �������
* @param minfilt_size ������ �������� ������������ �������
*/
cv::Vec3b get_atm_light(const cv::Mat& img, double stopdiv_size = 0.2, int minfilt_size = 5);

/* ���������� ����������� ����
* @param tmap ����� �����������
* @param img ������� BGR �����������
* @param gray ������� ����� �����������
* @param atm_light ����������� ������������ �����
* @param lambda �����������
*/
cv::Mat adaptive_atm_light(const cv::Mat& tmap, const cv::Mat& img, const cv::Mat& gray, cv::Scalar atm_light, int lambda);