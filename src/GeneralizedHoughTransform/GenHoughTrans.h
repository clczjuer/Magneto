#ifndef LP_GenHoughTrans_H_
#define LP_GenHoughTrans_H_

#include "cv.h"
#include "highgui.h"
#include "core\core.hpp"

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
	std::vector<Magneto::Rpoint2> pts2;

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

	int method;
public:
	enum EM_GHT_METHOD{
		emXY = 0x01, emRotate = 0x02, emScale = 0x04
	};
	GenHoughTrans();
	~GenHoughTrans();
	void setMethod(int method);
	int getIntervals() const { return intervals; }
	float getMinPhi() const { return phimin; }
	float getMaxPhi() const { return phimax; }
	int getRangeXY() const { return rangeXY; }

	void genRefPoint(cv::Mat &edge);
	void genRefPoint(cv::Point pt) {
		refPoint = pt; 
	}
	static void phase(cv::Mat &src, cv::Mat &angle);
	static void getEdgeInfo(cv::Mat &src, cv::Mat & edge, cv::Point pt, std::vector<Magneto::Rpoint> &pts);
	static void getEdgeInfo(cv::Mat &src, cv::Mat & edge, int intervals, float rangeXY, std::vector<Magneto::Rpoint2> &pts2);

	void createRTable(cv::Mat &src, cv::Mat & edge);

	void accumlate(cv::Mat & src, cv::Mat & edge);
	void accumlate4Shift();

	void accumlate4Shift(cv::Mat & src, cv::Mat & edge);
	void accumlate4ShiftAndRotate(cv::Mat & src, cv::Mat & edge);
	void accumlate4Rotate(cv::Mat & src, cv::Mat & edge);

	void bestCandidate(cv::Mat & src);

	void detect(cv::Size size, int r);

	static void RotateTransform(std::vector<std::vector<cv::Vec2i>> & RtableSrc, 
		std::vector<std::vector<cv::Vec2i>> &RtableRotate, int angleIndex, float deltaAngle);
	static void ScaleTransform(std::vector<std::vector<cv::Vec2i>> & RtableSrc, std::vector<std::vector<cv::Vec2i>> &RtableScaled, float dScale);
};



class GHTInvoker : public cv::ParallelLoopBody {
public:
	GHTInvoker(GenHoughTrans *_ght,cv::Size _size){
		ght = _ght;
		size = _size;
	}

	cv::Size size;
	GenHoughTrans *ght;

	virtual void operator() (const cv::Range& range) const;
};



#endif // LP_GenHoughTrans_H_

