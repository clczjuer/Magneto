#pragma once
#include "cv.h"
#include "highgui.h"
class LineSegmentDetector
{
public:
	LineSegmentDetector();
	~LineSegmentDetector();
	bool detect(cv::Mat &img, cv::Mat &result);
};

