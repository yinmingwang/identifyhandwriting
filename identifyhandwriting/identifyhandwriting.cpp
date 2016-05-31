#include "cv.h"
#include "highgui.h"
#include "iostream"
#include "string"
#include "cmath" 
#define PI 3.1415926
using namespace std;
using namespace cv;
struct Lines{
	double k;
	double b;
	bool per;
};
double rotationAngle = 0.0;
bool LineCmp(const Vec2f& a, const Vec2f& b) {
	double threshold = 0.1;
	if (abs(a[1] - b[1]) < threshold) {
		return a[0] < b[0];
	}
	return a[1] < b[1];
}

bool PointCmp(const Point2f& a, const Point2f& b) {
	return a.x + a.y < b.x + b.y;
}
vector<Lines> Findlines(Mat img) {
	vector<Vec2f> lines;
	vector<Lines> mls;
	//rotationAngle = 0.0;
	HoughLines(img, lines, 1, CV_PI / 100, 80, 0, 0);
	sort(lines.begin(), lines.end(), LineCmp);
	int last = lines.size() - 1;
	const double minTheta = 0.1;
	const double maxTheta = 3.1;
	const double rhoThreshold = 50;

	while (lines[0][1] <= minTheta && lines[last][1] >= maxTheta) {
		vector<Vec2f>::iterator it = lines.begin() + 1;
		for (size_t i = 1; i < last; ++i) {
			if (lines[i][1] <= minTheta && abs(lines[0][0] - lines[i][0]) < rhoThreshold) {
				++it;
			}
		}
		lines.insert(it, lines[last]);
		it = lines.end() - 1;
		lines.erase(it);
	}

	rotationAngle = lines[0][1] * 180 / CV_PI;
	if (rotationAngle < -10) {
		rotationAngle += 180;
	}
	const double rotationThreshold = 0.1;
	if (lines[0][1] > rotationThreshold) {
		rotationAngle -= 90;
	}
	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		const float deltaRho = 200;
		const float deltaTheta = 0.25;
		if (i > 0 && abs(abs(rho) - abs(lines[i - 1][0])) < deltaRho &&
			(abs(CV_PI - (theta + lines[i - 1][1])) < deltaTheta || abs(theta - lines[i - 1][1]) < deltaTheta)) {
			vector<Vec2f>::iterator it = lines.begin() + i;
			lines.erase(it);
			--i;
			continue;
		}
		if (i > 0 && i < lines.size() - 1 &&
			abs(theta - lines[i - 1][1]) > deltaTheta && abs(theta - lines[i + 1][1]) > deltaTheta &&
			abs(CV_PI - (theta + lines[i - 1][1])) > deltaTheta) {
			vector<Vec2f>::iterator it = lines.begin() + i;
			lines.erase(it);
			--i;
			continue;
		}
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(img, pt1, pt2, Scalar(255, 255, 255), 1, CV_AA);
		struct Lines ml;
		if (b == 0) {
			ml.per = 1;
			ml.b = rho;
		}
		else {
			ml.per = 0;
			double m = -a / b, c = rho / b;
			ml.k = m;
			ml.b = c;
		}
		mls.push_back(ml);
	}
	return mls;
}

Mat get_edge(Mat img) {
	Mat dstimg, blurImg;
	blur(img, blurImg, Size(3, 3));
	Canny(blurImg, dstimg, 130, 550, 3);
	return dstimg;
}


vector<Point2f> findintersection(vector<Lines> mls, Mat image) {
	vector<Point2f> points;
	for (int i = 0; i < 2; ++i) {
		for (int j = 2; j < 4; ++j) {
			double a = mls[i].k, b = mls[j].k;
			double c = mls[i].b, d = mls[j].b;
			double x, y;
			if (mls[i].per == 1) {
				x = c;
				y = b * c + d;
			}
			else {
				x = (d - c) / (a - b);
				y = (a * d - b * c) / (a - b);
			}
			//printf("Point: (%lf, %lf)\n", x, y);
			Point2f pt1;
			pt1.x = x;
			pt1.y = y;
			points.push_back(pt1);
			circle(image, pt1, 3, Scalar(255, 0, 0), 1);
		}
	}
	return points;
}
Mat CorrectImg(vector<Point2f> points, Mat img, Mat tempimg) {
	Point centerpoint;
	centerpoint.x = (points[0].x + points[1].x + points[2].x + points[3].x) / 4;
	centerpoint.y = (points[0].y + points[1].y + points[2].y + points[3].y) / 4;
	//circle(img, centerpoint, 3, Scalar(255, 255, 255), 1);
	Mat rotatedImg;
	Mat rotationMatrix;
	rotationMatrix = getRotationMatrix2D(centerpoint, rotationAngle, 1);
	warpAffine(tempimg, rotatedImg, rotationMatrix, rotatedImg.size());
	transform(points, points, rotationMatrix);
	if (tempimg.rows < tempimg.cols) {
		const float scale = 0.75;
		rotationMatrix = getRotationMatrix2D(centerpoint, 90, scale);
		warpAffine(rotatedImg, rotatedImg, rotationMatrix, rotatedImg.size());
		transform(points, points, rotationMatrix);
	}
	sort(points.begin(), points.end(), PointCmp);
	const int lx = 100;
	const int ly = 110;
	const int rx = 500;
	const int ry = 700;
	const int width = rx - lx;
	const int height = ry - ly;
	Point2f TopLeft;
	Point2f TopRight;
	Point2f BottomLeft;
	Point2f BottomRight;
	vector<Point2f> standardPoints;
	TopLeft.x = lx;
	TopLeft.y = ly;
	TopRight.x = rx;
	TopRight.y = ly;
	BottomLeft.x = lx;
	BottomLeft.y = ry;
	BottomRight.x = rx;
	BottomRight.y = ry;
	standardPoints.push_back(TopLeft);
	standardPoints.push_back(TopRight);
	standardPoints.push_back(BottomLeft);
	standardPoints.push_back(BottomRight);
	Mat perspectiveMatrix;
	perspectiveMatrix = getPerspectiveTransform(points, standardPoints);
	Mat perspectiveImg(800, 800, rotatedImg.type());
	warpPerspective(rotatedImg, perspectiveImg, perspectiveMatrix, perspectiveImg.size());
	Rect myROI(lx, ly, width, height);
	Mat CorrectImage = perspectiveImg(myROI);
	imshow("CorrectImage", CorrectImage);
	imwrite("4.jpg", CorrectImage);
	return CorrectImage;
}
int otsu(Mat img) {
	int histogram[256] = { 0 };
	double probability[256] = { 0.0 };
	/*计算灰度直方图*/
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			histogram[img.at<uchar>(i, j)]++;
		}
	}
	/*求出每个灰度值出现的比例*/
	for (int i = 0; i < 256; i++) {
		probability[i] = histogram[i] / ((img.rows*img.cols) + 0.0);
	}
	double averagelevel = 0.0;
	/*求出总的平均灰度*/
	for (int i = 0; i < 256; i++) {
		averagelevel += i*probability[i];
	}

	double max = 0.0;
	int maxthreshold = 0;
	for (int i = 0; i < 256; i++) {
		double p1 = 0.0;
		double p2 = 0.0;
		double averageuj = 0.0;
		/*计算前景的比例和平均灰度*/
		for (int j = 0; j <= i; j++) {
			p1 += probability[j];  //前景点数占图像比例
			averageuj += j*probability[j];
		}
		double m1 = averageuj / p1;//前景的平均灰度
		p2 = 1 - p1; //背景点数占图像比例
		double m2 = (averagelevel - averageuj) / p2;//背景的平均灰度
		double g = p1*p2*(m1 - m2)*(m1 - m2); //间内方差
		//cout << g << endl;
		if (g > max) {
			max = g; //求出最大间内方差
			maxthreshold = i; //求出使得间内方差最大的阈值
		}
	}
	//cout << maxthreshold << endl;
	return maxthreshold;
}
Mat getotsuimg(Mat srcimg) {
	Mat img;
	cvtColor(srcimg, img, CV_RGB2GRAY);
	//绘制直方图
	MatND hist;
	int nbins = 256;
	int hsize[] = { nbins };
	float range[] = { 0, 256 };
	const float* ranges[] = { range };
	calcHist(&img, 1, 0, Mat(), hist, 1, hsize, ranges);
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / nbins);
	Mat histImg(hist_h, hist_w, CV_32FC3, Scalar(0, 0, 0));
	normalize(hist, hist, 0, histImg.rows, NORM_MINMAX, -1, Mat());
	for (int i = 1; i < nbins; ++i) {
		line(histImg, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 255, 255), 2, 8, 0);
	}

	int threshold = otsu(img);
	string namehist = "imghist";
	line(histImg, Point(bin_w * threshold, 0), Point(bin_w * threshold, hist_h), Scalar(0, 0, 255), 1, CV_AA);
	imshow(namehist, histImg); //绘制直方图
	//imwrite(histpath,histImg);
	/*利用求出的阈值二值化图像*/
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) > threshold-40) {
				img.at<uchar>(i, j) = 255;
			}
			else {
				img.at<uchar>(i, j) = 0;
			}
		}
	}
	imshow("img", img);
	cvWaitKey(0);
	return img;
}
void segmentation() {
	IplImage* Igray = 0;
	IplImage* It = 0;
	IplImage* Iat;
	Igray = cvLoadImage("4.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	It = cvCreateImage(cvSize(Igray->width, Igray->height), IPL_DEPTH_8U, 1);
	Iat = cvCreateImage(cvSize(Igray->width, Igray->height), IPL_DEPTH_8U, 1);
	cvThreshold(Igray, It, 150, 255, CV_THRESH_BINARY);
	cvAdaptiveThreshold(Igray, Iat, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 35);
	cvNamedWindow("adaptiveThresh", 1);
	cvShowImage("adaptiveThresh", Iat);
	cvWaitKey(0);
	cvReleaseImage(&Iat);
	cvDestroyWindow("adaptiveThresh");
}
int main()
{
	Mat srcImg = imread("2.jpg");
	Mat blurImg, grayImg, dstImg, tempImg, cornerImg;
	int width = srcImg.cols;
	int height = srcImg.rows;
	resize(srcImg, tempImg, Size(width, height), 0, 0, CV_INTER_LINEAR);
	imshow("SrcImg", tempImg);
	dstImg = get_edge(tempImg);
	Mat image = dstImg.clone();
	vector<Lines> lines = Findlines(image);
	vector<Point2f> points = findintersection(lines, image);
	Mat corrimg = CorrectImg(points, image, tempImg);
	//Mat otsuimg = getotsuimg(corrimg);
    segmentation();
	waitKey(0);
	return 0;
}