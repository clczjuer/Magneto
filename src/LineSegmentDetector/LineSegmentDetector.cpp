#include "LineSegmentDetector.h"
#include "lsd_1.6/lsd.h"
using namespace cv;

LineSegmentDetector::LineSegmentDetector()
{
}


LineSegmentDetector::~LineSegmentDetector()
{
}

bool LineSegmentDetector::detect(cv::Mat &img, cv::Mat &result)
{
	double * image;
	double * out;
	int x, y, i, j, n;
	img(Rect(2440, 0, 1000, img.rows)).copyTo(img);
	medianBlur(img, img, 3);
	// resize(img, img, Size(img.cols/2, img.rows/2));
	//resize(img, img, Size(img.cols / 2, img.rows / 2));

	int X = img.cols;  /* x image size */
	int Y = img.rows;  /* y image size */

	/* create a simple image: left half black, right half gray */
	image = (double *)malloc(X * Y * sizeof(double));
	if (image == NULL)
	{
		fprintf(stderr, "error: not enough memory\n");
		exit(EXIT_FAILURE);
	}
	for (x = 0; x < X; x++)
		for (y = 0; y < Y; y++)
			image[x + y*X] = img.at<uchar>(y, x);// x < X / 2 ? 0.0 : 64.0; /* image(x,y) */


	/* LSD call */
	out = lsd(&n, image, X, Y);

	/* print output */
	printf("%d line segments found:\n", n);
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < 7; j++)
			printf("%f ", out[7 * i + j]);
		printf("\n");
	}

	cv::Mat imgColor;
	cv::cvtColor(img, imgColor, CV_GRAY2BGR);
	for (int i = 0; i < n; i++)
	{
		double *pData = &out[7 * i];
		for (int j = 0; j < 7; j++)
		{
			Point pt1(pData[0], pData[1]);
			Point pt2(pData[2], pData[3]);

			line(imgColor, pt1, pt2, Scalar(255, 0, 0), 2);
		}

	}

	/* free memory */
	free((void *)image);
	free((void *)out);

	return true;
}