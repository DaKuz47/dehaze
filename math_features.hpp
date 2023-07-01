#pragma once

#include <vector>
#include <cmath>
#include <iostream>

// ��������� ���� �������� ������� (������ �������)
std::vector<double> gaussianKernel(double sigma);

// ��������� ����������� ������������� (��������)
double gauss(double x, double sigma);

// ���������� ��������� (�������� �����)
double marra(double x, double sigma);
