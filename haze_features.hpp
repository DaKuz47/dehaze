#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


 
/* �������, ������������ ����������� ������� ��� ��� �������� ������� �����������: 
������, ���� - ������, ������ - �������
* @param image - ������� BGR �����������
*/
std::vector<cv::Mat> contrastEnergy(const cv::Mat& image);

/* �������, ������������ �������� �����������, ��������� ����� ����� �����������
* @param gray ������� ����� ����� �����������
* @param patch_size ������ ���������� �������
*/
double imageEntropy(const cv::Mat& gray, int patch_size);

/* �������, ������� ����������� ����������
* @param gray ������� ����� ����� �����������
* @param gaussKernel ���� ������� ������
* @param mu ��
*/
double stdDeviation(const cv::Mat& gray, const cv::Mat& gaussKernel, double mu);

/* �������, ������� ��������������� ���������
*/
double normDisp(double std, double mu);

/* ����� ����������� ����� �����������
* @param patches ������� ����������� ���������� �� ������ � ���� ��������
* @param a ���������� ������������ �����
* @param patch_size ������ �����
* @param max_iter ������������ ���������� �������� � ��������� ������� ����
* @param eps �������� � ��������� ������� ����
* @param log ����� ��������
*/
cv::Mat tmap_optimal(cv::Mat& patches, cv::Scalar a, int patch_size, int max_iter, double eps, bool log);