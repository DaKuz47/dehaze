#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


/* ����������� ������. ��������� ������������� ����� ��������. 
������������ ��� ������������� ��� ���������� ����������� ������������
* @param img ������� �����������
* @param out_img �������� �����������
* @param size ������ ��������
*/
void min_filter(const cv::Mat& img, cv::Mat& out_img, int size);

/* GuidedFilter �������� ����� �����������, �������� �� ��������� �����������
* @param img ������� �����������
* @param p ����� �����������
* @param r �����������
* @param eps ��������
*/
cv::Mat guided_filter(const cv::Mat& img, const cv::Mat& p, int r, double eps);