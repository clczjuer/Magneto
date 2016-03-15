#ifndef LP_Manipulation_h__
#define LP_Manipulation_h__

#include "cv.h"

namespace Magneto {
	
	template <typename T>
	inline T* ptrat1D(cv::Mat &mt, int i0){
		return (T*)(mt.data + i0*mt.step.p[0]);
	}

	template <typename T>
	inline T* ptrat2D(cv::Mat &mt, int i0, int i1){
		return (T*)(mt.data + i0*mt.step.p[0] + i1*mt.step.p[1]);
	}

	template <typename T>
	inline T* ptrat3D(cv::Mat &mt, int i0, int i1, int i2){
		return (T*)(mt.data + i0*mt.step.p[0] + i1*mt.step.p[1] + i2*mt.step.p[2]);
	}

	template <typename T>
	inline T* ptrat4D(cv::Mat &mt, int i0, int i1, int i2, int i3){
		return (T*)(mt.data + i0*mt.step.p[0] + i1*mt.step.p[1] + i2*mt.step.p[2] + i3*mt.step.p[3]);
	}

	inline int roundToInt(float num) {
		return (num > 0.0) ? (int)(num + 0.5f) : (int)(num - 0.5f);
	}
}
#endif // LP_Manipulation_h__