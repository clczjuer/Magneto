
#include <QtCore/QCoreApplication>
#include "GenHoughTrans.h"

int main(int argc, char *argv[])
{
	QCoreApplication a(argc, argv);

	GenHoughTrans ght;
	QString strPath = qApp->applicationDirPath();
	cv::Mat imgTemplate = cv::imread(QString(strPath + "//files//contour_def.bmp").toLatin1().data(), -1);
	ght.genRefPoint(imgTemplate);
	cv::Mat imgSrc = cv::imread(QString(strPath + "//files//template_original.jpg").toLatin1().data(), 0);
	ght.createRTable(imgSrc, imgTemplate);
	cv::Mat imgEdge = cv::imread(QString(strPath + "//files//contour_rough.bmp").toLatin1().data(), 0);
	ght.accumlate4Shift(imgSrc, imgEdge);
	ght.bestCandidate(imgSrc);

	return a.exec();
}
