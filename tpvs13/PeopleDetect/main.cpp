
#include <QtCore/QCoreApplication>
#include "cv.h"
#include "highgui.h"

using namespace cv;

int main(int argc, char *argv[])
{
	QCoreApplication a(argc, argv);

	std::vector<Rect> found, found_filtered;
	cv::HOGDescriptor people_dectect_hog;
	//����Ĭ�ϵ��Ѿ�ѵ�����˵�svmϵ����Ϊ�˴μ���ģ��
	people_dectect_hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
	//�������ͼƬimg���ж�߶����˼��
	//imgΪ���������ͼƬ��foundΪ��⵽Ŀ�������б�����3Ϊ�����ڲ�����Ϊ����Ŀ�����ֵ��Ҳ���Ǽ�⵽��������SVM���೬ƽ��ľ���;
	//����4Ϊ��������ÿ���ƶ��ľ��롣�������ǿ��ƶ���������������5Ϊͼ������Ĵ�С������6Ϊ����ϵ����������ͼƬÿ�γߴ��������ӵı�����
	//����7Ϊ����ֵ����У��ϵ������һ��Ŀ�걻������ڼ�����ʱ���ò�����ʱ�����˵������ã�Ϊ0ʱ��ʾ����������á�
	QString strPath = qApp->applicationDirPath();
	cv::Mat img = cv::imread(QString(strPath+"//files//PeopleDetect//test.jpg").toLatin1().data());
	if (img.empty()) {
		return 1;
	}
	people_dectect_hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
	//��Դ���п��Կ���:
	//#define __SIZE_TYPE__ long unsigned int
	//typedef __SIZE_TYPE__ size_t;
	//���,size_t��һ��long unsigned int����
	size_t i, j;
	for (i = 0; i < found.size(); i++)
	{
		Rect r = found[i];

		//��������for������ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵�
		//��,��ȡ���������Ǹ����ο����found_filtered��
		for (j = 0; j < found.size(); j++)
			if (j != i && (r&found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);
	}

	//��ͼƬimg�ϻ������ο�,��Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ
	//��һЩ����
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
