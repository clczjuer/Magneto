
#include <QtCore/QCoreApplication>
#include "cv.h"
#include "highgui.h"

using namespace cv;

int main(int argc, char *argv[])
{
	QCoreApplication a(argc, argv);

	std::vector<Rect> found, found_filtered;
	cv::HOGDescriptor people_dectect_hog;
	//采用默认的已经训练好了的svm系数作为此次检测的模型
	people_dectect_hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
	//对输入的图片img进行多尺度行人检测
	//img为输入待检测的图片；found为检测到目标区域列表；参数3为程序内部计算为行人目标的阈值，也就是检测到的特征到SVM分类超平面的距离;
	//参数4为滑动窗口每次移动的距离。它必须是块移动的整数倍；参数5为图像扩充的大小；参数6为比例系数，即测试图片每次尺寸缩放增加的比例；
	//参数7为组阈值，即校正系数，当一个目标被多个窗口检测出来时，该参数此时就起了调节作用，为0时表示不起调节作用。
	QString strPath = qApp->applicationDirPath();
	cv::Mat img = cv::imread(QString(strPath+"//files//PeopleDetect//test.jpg").toLatin1().data());
	if (img.empty()) {
		return 1;
	}
	people_dectect_hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
	//从源码中可以看出:
	//#define __SIZE_TYPE__ long unsigned int
	//typedef __SIZE_TYPE__ size_t;
	//因此,size_t是一个long unsigned int类型
	size_t i, j;
	for (i = 0; i < found.size(); i++)
	{
		Rect r = found[i];

		//下面的这个for语句是找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的
		//话,则取外面最大的那个矩形框放入found_filtered中
		for (j = 0; j < found.size(); j++)
			if (j != i && (r&found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);
	}

	//在图片img上画出矩形框,因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要
	//做一些调整
	for (i = 0; i < found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		rectangle(img, r.tl(), r.br(), Scalar(0, 255, 0), 3);
	}
	namedWindow("show");
	imshow("show", img);
	waitKey(0);

	return a.exec();
}
