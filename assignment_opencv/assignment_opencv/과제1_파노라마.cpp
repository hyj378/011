//Ư¡�� ����->���->¦�� �� �̹��� ã��, �̹��� ��ȯ->����
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;
//�̹������� Ư¡�� ã�� �Լ�
vector<KeyPoint> findkeypoint(Mat img)
{
	//�̹������� Ư¡�� ã�� (SIFT��)
	vector<KeyPoint> arrayKey;
	double nMinHessian = 400.;
	Ptr<xfeatures2d::SiftFeatureDetector> Detector = xfeatures2d::SiftFeatureDetector::create(nMinHessian);

	Detector->detect(img, arrayKey);
	return arrayKey;
}
//�� �ٸ� �ĳ�� �ڵ�
//https://stackoverflow.com/questions/23492878/image-disappear-in-panorama-opencv
Mat deleteBlackZone(const Mat &image)
{
	Mat resultGray;
	Mat result;
	image.copyTo(result);

	cvtColor(image, resultGray, CV_RGB2GRAY);

	medianBlur(resultGray, resultGray, 3);
	// ù��° �Ű����� : ���� �̹���, �ι�° : ���͸� ��ģ �̹��� ����° : ���������� ������
	Mat resultTh;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	threshold(resultGray, resultTh, 1, 255, 0);
	findContours(resultTh, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	//���� ������ ���� ������Ʈ�� �ܰ����� �����ϴ� �Լ�
	//CV_RETR_EXTERNAL : �ܺ� �ܰ��� �˻�
	//CV_CHAIN_APPROX_SIMPLE : ������ ���� ���� �Ǵ� ����, �밢�� �ܰ����� ���Ե�

	Mat besta = Mat(contours[0]);

	Rect a = boundingRect(besta);
	cv::Mat half(result, a);
	return half;

}
//�̹��� 2�� ���̰� ���� ������ return
Mat addtwo(Mat matbody, Mat matarm) 
{
	//imshow("startfrombody", matbody);
	//imshow("startfromarm", matarm);

	//��� �ӵ��� ���� �������
	Mat matgraybody, matgrayarm;
	cvtColor(matbody, matgraybody, CV_RGB2GRAY);
	cvtColor(matarm, matgrayarm, CV_RGB2GRAY);

	//�� �̹������� keypoint ����
	vector<KeyPoint> keyPointbody, keyPointarm;
	keyPointbody = findkeypoint(matgraybody);
	keyPointarm = findkeypoint(matgrayarm);

	//�� �̹������� Ű����Ʈ�� ����ϱ�
	Ptr<xfeatures2d::SiftDescriptorExtractor> Extractor = xfeatures2d::SiftDescriptorExtractor::create();
	Mat matExtractorbody, matExtractorarm;
	Extractor->compute(matbody, keyPointbody, matExtractorbody);
	Extractor->compute(matarm, keyPointarm, matExtractorarm);

	//Ű����Ʈ�� ��Ī�ϰ� ��Ī���� DMatch�� ���� �� good_match�� ��󳽴�.
	//ransac�� badmatch�� �ɷ��ִ°� �ƴѰ�?? �� goodmatch�� �������� �ٽ�����?
	FlannBasedMatcher Matcher;
	vector<DMatch> matches;
		//cv::DescriptorMatcher::match(����, Ʈ����)
	Matcher.match(matExtractorbody, matExtractorarm, matches);

	//�ŷ��� �� �ִ� ��ġ ��󳻱�
	double dMinDist = 100;
	double dMaxDist = 0;
	for (int i = 0; i < matches.size(); i++)
	{
		double dDistance = matches[i].distance;

		if (dDistance < dMinDist) dMinDist = dDistance;
		if (dDistance > dMaxDist) dMaxDist = dDistance;
	}

	printf("-- Max iddst : %f \n", dMaxDist);
	printf("-- Min iddst : %f \n", dMinDist);
	vector<DMatch> good_matches;
	for (int i = 0; i < matches.size(); i++)
	{
		if (matches[i].distance < dMinDist * 5)
		{
			good_matches.push_back(matches[i]);
		}
	}

	Mat matGoodMatches1;
	drawMatches(matbody, keyPointbody, matarm, keyPointarm, good_matches, matGoodMatches1, Scalar::all(-1), Scalar(-1), vector<char>(), DrawMatchesFlags::DEFAULT);
	imshow("all-matches", matGoodMatches1);

	//�ŷ��� �� �ִ� ��Ī�� good_match�� �̿��� ��ȯ����� ã��
	//findhomography�� ransac�˰������� ��ȯ����� ã���ش�
	//��ȯ����� shape�� ���� �ñ�... ��ȯ���� if��ȯ���� �������ִٸ� ��� ��ȯapi�� ������
	//mat�� (2*2, 3*3, 4*4)�� ����? findhomography�� �μ��� point�� ������ Dmatch�� point�� ��ȯ
	vector<Point2f> pointsbody, pointsarm;
	for (int i = 0; i < good_matches.size(); i++) 
	{
		pointsbody.push_back(keyPointbody[good_matches[i].queryIdx].pt);
		pointsarm.push_back(keyPointarm[good_matches[i].trainIdx].pt);
	}
	Mat HomoMatrix = findHomography(pointsarm, pointsbody, CV_RANSAC);
	cout << HomoMatrix << endl;
	//*******************************************************************
	Mat matarmResult;
	warpPerspective(matarm, matarmResult, HomoMatrix, Size(matarm.cols * 2, matarm.rows), INTER_CUBIC);
	imshow("warp", matarmResult);

	Mat matPanorama = matarmResult.clone();
	//http://visionprogrammer.tistory.com/8
	//ROI�����̶� ���ɿ����̶�� ���̴�.
	Mat matROI(matPanorama, Rect(0, 0, matbody.cols, matbody.rows));
	matbody.copyTo(matROI);
	
	imshow("panorama", matPanorama);
	waitKey(0);

	//std::vector<int> compression_params;
	//imshow("panorama", matPanorama);

	return matPanorama;
}
int main()
{
	//int num = 0;
	//std::cout << "�Է��� ������ ������?(3�Է�)" << std::endl;
	//std::cin >> num; //���߿� string���� �ּҵ� �Է¹޴°ɷ�

	Mat CMatLimg;//����
	Mat CMatMimg;//�߰�
	Mat CMatRimg;//����

	//�̹��� �ε�
	CMatLimg = imread("C:/Users/Ȳ����/Pictures/L.jpg", IMREAD_COLOR);
	CMatMimg = imread("C:/Users/Ȳ����/Pictures/M.jpg", IMREAD_COLOR);
	CMatRimg = imread("C:/Users/Ȳ����/Pictures/R.jpg", IMREAD_COLOR);

	if (CMatLimg.empty() || CMatMimg.empty() || CMatRimg.empty())
	{
		cout << "image load fail" << endl;
		return -1;
	}

	//ũ��� �׳� ���� ������ ������...
	Size size(CMatLimg.cols / 2, CMatLimg.rows / 2);
	resize(CMatLimg, CMatLimg, size);
	resize(CMatMimg, CMatMimg, size);
	resize(CMatRimg, CMatRimg, size);

	Mat matpanorama;
	//Mat plus(Mat matbody, Mat matarm) �տ��� body
	matpanorama = addtwo(CMatLimg, CMatMimg);
	matpanorama=deleteBlackZone(matpanorama);
	matpanorama = addtwo(matpanorama, CMatRimg);

	imshow("final", matpanorama);
	waitKey(0);

	return 0;
}