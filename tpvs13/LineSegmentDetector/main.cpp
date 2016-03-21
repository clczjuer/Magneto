
#include <QtCore/QCoreApplication>
#include "LineSegmentDetector.h"
int main(int argc, char *argv[])
{
	QCoreApplication a(argc, argv);
	QString strPath = qApp->applicationDirPath();
	cv::Mat img = cv::imread("c://°×Ä¤ºáÎÆ6.BMP", 0);
	cv::Mat result;
	LineSegmentDetector detector;
	detector.detect(img, result);
	return a.exec();
}
