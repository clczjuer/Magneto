#ifndef LP_GenHoughTrans_H_
#define LP_GenHoughTrans_H_

#include "cv.h"
#include "highgui.h"

namespace Magneto {
	struct Rpoint {
		int dx;
		int dy;
		float phi;
	};

	struct Rpoint2 {
		float x;
		float y;
		int phiIndex;
	};

}

class GenHoughTrans
{
	cv::Mat accum;
	std::vector<Magneto::Rpoint> pts;

	// reference point (inside contour)
	cv::Vec2i refPoint;
	// R-table of template object:
	std::vector<std::vector<cv::Vec2i>> Rtable;
	// number of intervals for angles of R-table:
	int intervals;
	// minimum and maximum rotation allowed for template
	float phimin;
	float phimax;
	// dimension in pixels of squares in image
	int rangeXY;
public:
	GenHoughTrans();
	~GenHoughTrans();

	void genRefPoint(cv::Mat &edge);
	void setRefPoint(cv::Point pt) { 
		refPoint = pt; 
	}
	static void phase(cv::Mat &src, cv::Mat &angle);
	static void getEdgeInfo(cv::Mat &src, cv::Mat & edge, cv::Point pt, std::vector<Magneto::Rpoint> &pts);
	static void getEdgeInfo(cv::Mat &src, cv::Mat & edge, int intervals, float rangeXY, std::vector<Magneto::Rpoint2> &pts2);

	void createRTable(cv::Mat &src, cv::Mat & edge);

	void accumlate4Shift(cv::Mat & src, cv::Mat & edge);
	void accumlate4ShiftAndRotate(cv::Mat & src, cv::Mat & edge);

	void bestCandidate(cv::Mat & src);

};
#endif // LP_GenHoughTrans_H_

