#include "cv.h"
#include "highgui.h"
#include "iostream"
#include "string"
#include "cmath"
#include <sstream>
#define PI 3.1415926
using namespace std;
using namespace cv;
RNG rng(12345);
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
	Mat ReImg(CorrectImage, Rect(10, 10, CorrectImage.cols - 20, CorrectImage.rows-20));
	//imshow("CorrectImage", CorrectImage);
	return ReImg;
}

Mat segmentation(Mat srcimg) {
	Mat grayimg;
	Mat seimg;
	cvtColor(srcimg, grayimg, CV_BGR2GRAY);
	adaptiveThreshold(grayimg, seimg, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 33, 5);
	//imshow("seimg", seimg);
	return seimg;
}
Mat rotation(Mat srcImg, double angle) {
	Mat tempImg;
	CV_Assert(!srcImg.empty());
	float radian = (float)(angle / 180.0 * CV_PI);
	//填充图像使其符合旋转要求
	int uniSize = (int)(max(srcImg.cols, srcImg.rows)* 1.414);
	int dx = (int)(uniSize - srcImg.cols) / 2;
	int dy = (int)(uniSize - srcImg.rows) / 2;
	copyMakeBorder(srcImg, tempImg, dy, dy, dx, dx, BORDER_CONSTANT);
	//旋转中心
	Point2f center((float)(tempImg.cols / 2), (float)(tempImg.rows / 2));
	Mat affine_matrix = getRotationMatrix2D(center, angle, 1.0);
	//旋转
	warpAffine(tempImg, tempImg, affine_matrix, tempImg.size());
	//旋转后的图像大小
	float sinVal = fabs(sin(radian));
	float cosVal = fabs(cos(radian));
	Size targetSize((int)(srcImg.cols * cosVal + srcImg.rows * sinVal),
		(int)(srcImg.cols * sinVal + srcImg.rows * cosVal));

	//剪掉四周边框
	int x = (tempImg.cols - targetSize.width) / 2;
	int y = (tempImg.rows - targetSize.height) / 2;
	Rect rect(x, y, targetSize.width, targetSize.height);
	tempImg = Mat(tempImg, rect);
	//imshow("Show", tempImg);
	return tempImg;
}
Mat Erosion(Mat srcimg,int size) {
	Mat erodeimg;
	Mat element = getStructuringElement(MORPH_RECT, Size(size, size));
	/// 腐蚀操作
	erode(srcimg, erodeimg, element);
	//imshow("Erosion Demo", erodeimg);
	return erodeimg;
}
Mat Dilation(Mat srcimg,int size)
{
	Mat dilatimg;
	Mat element = getStructuringElement(MORPH_RECT, Size(size, size));
	/// 膨胀操作
	dilate(srcimg, dilatimg, element);
	//imshow("Dilation Demo", dilatimg);
	return dilatimg;
}
Mat Slicimg(Mat srcimg,int cmin,int cmax) {
	Mat threshold_output = srcimg.clone();
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<vector<Point>>::const_iterator itc = contours.begin();
	while (itc != contours.end()) {

		if (itc->size() < cmin || itc->size() > cmax)
			itc = contours.erase(itc);
		else
			++itc;
	}


	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}


	/// 画多边形轮廓 + 包围的矩形框
	//Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	Mat drawing = srcimg.clone();
	Mat src = srcimg.clone();
	vector<Mat> roi(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		stringstream stream;
		string str;
		stream << i;
		stream >> str;
		Scalar color = Scalar(255, 255, 255);
		drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 1, 8, 0);
		Mat temp(src, Rect(boundRect[i]));
		imwrite("./result/"+str+".jpg", temp);
	}
	return drawing;
}
int main()
{
	Mat srcImg = imread("1.jpg");
	Mat blurImg, grayImg, dstImg, tempImg, cornerImg;
	int width = srcImg.cols;
	int height = srcImg.rows;
	resize(srcImg, tempImg, Size(width, height), 0, 0, CV_INTER_LINEAR);
	//imshow("SrcImg", tempImg);
	dstImg = get_edge(tempImg);
	Mat image = dstImg.clone();
	vector<Lines> lines = Findlines(image);
	vector<Point2f> points = findintersection(lines, image);
	Mat corrimg = CorrectImg(points, image, tempImg);
	Mat seImg = segmentation(corrimg);
	Mat roimg = rotation(seImg, -90);
	Mat dilaimg = Dilation(roimg, 4);
	imshow("roimg", dilaimg);
	Mat silimg = Slicimg(dilaimg, 27, 200);
	imshow("roimg", silimg);
	waitKey(0);
	return 0;
}