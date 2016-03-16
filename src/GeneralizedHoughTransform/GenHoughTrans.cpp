#include "GenHoughTrans.h"
#include "Manipulation.h"

using namespace cv;
using namespace Magneto;

GenHoughTrans::GenHoughTrans()
{
	intervals = 36;
	phimin = -CV_PI;
	phimax = CV_PI;
	rangeXY = 1;
	method = emXY + emRotate + emScale;
	deltaPhi = CV_PI / intervals;
}

GenHoughTrans::~GenHoughTrans()
{
}

void GenHoughTrans::genRefPoint(cv::Mat &edge)
{
	for (int i = 0; i < edge.rows; i++) {
		for (int j = 0; j < edge.cols; j++) {
			if (edge.at<uchar>(i, j) == 127) {
				refPoint = Point(j, i);
				break;
			}
		}
	}
}

void GenHoughTrans::getEdgeInfo(cv::Mat &src, cv::Mat & edge, Point pt, std::vector<Rpoint> &pts)
{
	assert(src.size() == edge.size());
	Mat angle;
	GenHoughTrans::phase(src, angle);

	// contour points:
	pts.clear();
	int nr = src.rows;
	int nc = src.cols;
	for (int j = 0; j < nr; ++j) {
		uchar* pEdgeData = (uchar*)edge.ptr<uchar>(j);
		float* pAngleData = (float*)angle.ptr<float>(j);
		for (int i = 0; i < nc; ++i) {
			if (255 == pEdgeData[i])
			{
				Rpoint rpt;
				rpt.dx = pt.x - i;
				rpt.dy = pt.y - j;
				float a1 = pAngleData[i];
				rpt.phi = ((a1 > CV_PI) ? a1 - CV_PI : a1);

				// update further right and left dx
				//if (rpt.dx < mindx) mindx = rpt.dx;
				//if (rpt.dx > maxdx) maxdx = rpt.dx;
				pts.push_back(rpt);
			}
		}
	}
}

void GenHoughTrans::getEdgeInfo(cv::Mat &src, cv::Mat & edge, int intervals, float rangeXY, std::vector<Rpoint2> &pts2)
{
	int nr = src.rows;
	int nc = src.cols;
	float deltaphi = CV_PI / intervals;
	float inv_deltaphi = (float)intervals / CV_PI;
	float inv_rangeXY = (float)1 / rangeXY;
	cv::Mat angle;
	GenHoughTrans::phase(src, angle);

	for (int j = 0; j < nr; ++j) {
		uchar *pEdgeData = (uchar *)edge.ptr<uchar>(j);
		float *pAngleData = (float*)angle.ptr<float>(j);
		for (int i = 0; i < nc; ++i) {
			if (255 == pEdgeData[i]) // consider only white points (contour)
			{
				Rpoint2 rpt;
				rpt.x = i*inv_rangeXY;
				rpt.y = j*inv_rangeXY;
				//float a = atan2((float)vy, (float)vx);              //	gradient angle in radians
				float a = pAngleData[i];
				float phi = ((a > CV_PI) ? a - CV_PI : a);      // contour angle with respect to x axis
				int angleindex = (int)((phi)*inv_deltaphi); // index associated with angle (0 index = -90 degrees)
				if (angleindex == intervals) angleindex = intervals - 1;// -90 and +90 has same effect
				rpt.phiIndex = angleindex;
				pts2.push_back(rpt);
			}
		}
	}
}

void GenHoughTrans::phase(Mat &src, Mat &angle)
{
	if (src.empty()) {
		return;
	}
	Mat dx;
	dx.create(Size(src.cols, src.rows), CV_32F);
	Sobel(src, dx, CV_32F, 1, 0, CV_SCHARR);
	Mat dy;
	dy.create(Size(src.cols, src.rows), CV_32F);
	Sobel(src, dy, CV_32F, 0, 1, CV_SCHARR);
	cv::phase(dx, dy, angle);
}

void GenHoughTrans::createRTable(cv::Mat &src, cv::Mat & edge)
{
	std::vector<Rpoint> pts;
	GenHoughTrans::getEdgeInfo(src, edge, refPoint, pts);

	Rtable.clear();
	Rtable.resize(intervals);
	// put points in the right interval, according to discretized angle and range size 
	float range = CV_PI / intervals;
	for (vector<Rpoint>::size_type t = 0; t < pts.size(); ++t){
		int angleindex = (int)((pts[t].phi) / range);
		if (angleindex == intervals) angleindex = intervals - 1;
		Rtable[angleindex].push_back(Vec2i(pts[t].dx, pts[t].dy));
	}
}

void GenHoughTrans::setMethod(int method)
{
	this->method = method;
}

void GenHoughTrans::accumlate(cv::Mat & src, cv::Mat & edge)
{
	GenHoughTrans::getEdgeInfo(src, edge, intervals, rangeXY, pts2);

	int nr = src.rows;
	int nc = src.cols;
	Size size(nc, nr);

	int r0 = getFloorR();
	R = getCeilR() - getFloorR();
	int X = getX(nc);
	int Y = getY(nr);
	S = 1;

	int matSizep_S[] = { X, Y, R};
	accum.create(3, matSizep_S, CV_16S);
	accum = Scalar::all(0);

	double start = getTickCount();

	parallel_for_(Range(0, GenHoughTrans::genKey(getR(), getS())), GHTInvoker(this, Size(nc, nr)));

	double end = getTickCount();
	double time = (end - start) / getTickFrequency() * 1000.0;
	std::cout << "total" << ":   " << time << std::endl;
}

void GenHoughTrans::bestCandidate(cv::Mat & src)
{
	int id_min[4] = { 0, 0, 0, 0 };
	int id_max[4] = { 0, 0, 0, 0 };
	double minval;
	double maxval;
	minMaxIdx(accum, &minval, &maxval, id_min, id_max);
	int nr = src.rows;
	int nc = src.cols;

	// reference point
	Vec2i referenceP;
	if ( (method & emXY) == 0) {
		referenceP = refPoint;
	}
	else {
		referenceP = Vec2i(id_max[0] * getRangeXY() + (getRangeXY() + 1) / 2,
			id_max[1] * getRangeXY() + (getRangeXY() + 1) / 2);
	}

	// rotate index
	if ((emRotate & method) == 1) {
		double dMax = -1;
		int index = 0;
		for (int i = 0; i < nr; i++) {
			double r = (*ptrat3D<short>(accum, refPoint(0), refPoint(1), i));
			if (r > dMax) {
				index = i;
				dMax = r;
			}
		}
		id_max[2] = index;
	}

	int reff = id_max[2] + getFloorR();
	float cs = cos(reff*deltaPhi);
	float sn = sin(reff*deltaPhi);

	//! S
	float wratio = 1;// (float)w / (wtemplate);
	if ((emScale & method) == 1) {
		//int w = wmin + id_max[2] * rangeS;
		//float wratio = (float)w / (wtemplate);
	}
	
	Mat	input_img2;
	cvtColor(src, input_img2, CV_GRAY2BGR);
	cv::circle(input_img2, referenceP, 3, Scalar(0, 0, 255), 2);
	for (std::vector<std::vector<Vec2i>>::size_type ii = 0; ii < Rtable.size(); ++ii){
		for (std::vector<Vec2i>::size_type jj = 0; jj < Rtable[ii].size(); ++jj){
			int iimod = (ii + reff) % intervals;
			int dx = roundToInt(wratio*(cs*Rtable[ii][jj][0] - sn*Rtable[ii][jj][1]));
			int dy = roundToInt(wratio*(sn*Rtable[ii][jj][0] + cs*Rtable[ii][jj][1]));
			int x = referenceP[0] - dx;
			int y = referenceP[1] - dy;
			if ((x<nc) && (y<nr) && (x>-1) && (y>-1)){
				input_img2.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
			}
		}
	}
	// show result
	bool alt = false;
	for (;;)
	{
		char c = (char)waitKey(750);
		if (c == 27)
			break;
		if (alt){
			imshow("input_img", input_img2);
		}
		else{
			imshow("input_img", src);
		}
		alt = !alt;
	}
}

void GenHoughTrans::detect(Size size, int r, int s)
{
	double start = getTickCount();
	int nr = size.height;
	int nc = size.width;

	int r0 = getFloorR();
	int X = getX(nc);
	int Y = getY(nr);

	int reff = r + r0;
	std::vector<std::vector<Vec2f>> RtableTransform;

	if (1 == getS()) {
		GenHoughTrans::RotateTransform(Rtable, RtableTransform, reff, getDeltaPhi());
	}
	else  {
		std::vector<std::vector<Vec2f>> Rtablerotated;
		GenHoughTrans::RotateTransform(Rtable, Rtablerotated, reff, getDeltaPhi());
		GenHoughTrans::ScaleTransform(Rtablerotated, RtableTransform, s);
	}

	for (vector<Rpoint2>::size_type t = 0; t < pts2.size(); ++t){ // XY plane				
		int angleindex = pts2[t].phiIndex;
		int nSize = RtableTransform[angleindex].size();
		std::vector<Vec2f> * pVec = &(RtableTransform[angleindex]);
		for (std::vector<Vec2f>::size_type index = 0; index < nSize; ++index){
			float deltax = pVec->at(index)[0];
			float deltay = pVec->at(index)[1];
			int xcell = (int)(pts2[t].x + deltax);
			int ycell = (int)(pts2[t].y + deltay);
			if ((xcell<X) && (ycell<Y) && (xcell>-1) && (ycell>-1)){
				(*ptrat3D<short>(accum, xcell, ycell, r))++;
			}
		}
	}
	double end = getTickCount();
	double time = (end - start) / getTickFrequency() * 1000.0;
}

void GHTInvoker::operator()(const Range& range) const
{
	int i0 = range.start;
	int i1 = range.end;
	assert(abs(i1 - i0) == 1);
	int r, s;
	GenHoughTrans::parse(i0, ght->getR(), r, s);
	ght->detect(size, r, s);
}