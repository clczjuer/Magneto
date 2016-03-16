
#include <QtCore/QCoreApplication>
#include "GenHoughTrans.h"
using namespace cv;


int main(int argc, char *argv[])
{
	QCoreApplication a(argc, argv);

	GenHoughTrans ght;
	QString strPath = qApp->applicationDirPath();


#define LP_LOCK_1
#ifdef LP_LOCK_
	{
		cv::Mat imgTemplate = cv::imread(QString(strPath + "//files//contour_def.bmp").toLatin1().data(), -1);
		ght.genRefPoint(imgTemplate);
		cv::Mat imgSrc = cv::imread(QString(strPath + "//files//template_original.jpg").toLatin1().data(), 0);
		ght.createRTable(imgSrc, imgTemplate);
		cv::Mat imgEdge = cv::imread(QString(strPath + "//files//contour_rough.bmp").toLatin1().data(), 0);
		imgSrc = cv::imread(QString(strPath + "//files//rotate25.bmp").toLatin1().data(), 0);
		Canny(imgSrc, imgEdge, 80, 100, 3);
		//	ght.accumlate4Shift(imgSrc, imgEdge);
		ght.setMethod(GenHoughTrans::emRotate);
		ght.accumlate(imgSrc, imgEdge);
		ght.bestCandidate(imgSrc);
	}
#endif // LP_LOCK_

	{
		cv::Mat imgTemplate;
		cv::Mat imgSrc = cv::imread(QString(strPath + "//files//GeneralizedHoughTransform//src2.png").toLatin1().data(), 0);
		double dScale = 4;
		//resize(imgSrc, imgSrc, Size(imgSrc.cols / dScale, imgSrc.rows / dScale));

		cv::Point ptCenter(1464/dScale, 1000/dScale);
		ght.genRefPoint(ptCenter);

		Canny(imgSrc, imgTemplate, 200, 300);
		cv::Mat mask; imgSrc.copyTo(mask);
		mask.setTo(0);
		cv::circle(mask, ptCenter, 700/dScale, cv::Scalar(255), -1);
		cv::circle(mask, ptCenter, 400/dScale, cv::Scalar(0), -1);
		cv::min(mask, imgTemplate, imgTemplate);
		std::vector<std::vector<cv::Point>> vec;
		cv::findContours(imgTemplate, vec, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		imgTemplate.setTo(0);
		for (int i = 0; i < vec.size(); i++) {
			if (vec[i].size() < 70) {
				continue;
			}
			RotatedRect rt = minAreaRect(vec[i]);
// 			if (rt.size.height / rt.size.width < ) {
// 			
// 			}
			drawContours(imgTemplate, vec, i, Scalar(255), 1);
			//break;
		}

		ght.createRTable(imgSrc, imgTemplate);

		Mat imgEdge;
		Mat imgDst = cv::imread(QString(strPath + "//files//GeneralizedHoughTransform//dst2.png").toLatin1().data(), 0);
		//resize(imgDst, imgDst, Size(imgDst.cols / dScale, imgDst.rows / dScale));

		//imgSrc = cv::imread(QString(strPath + "//files//rotate25.bmp").toLatin1().data(), 0);
		Canny(imgDst, imgEdge, 200, 300);
		//	ght.accumlate4Shift(imgSrc, imgEdge);
		ght.setMethod(GenHoughTrans::emRotate);
		ght.accumlate(imgDst, imgEdge);
		ght.bestCandidate(imgDst);
	}
	return a.exec();
}
