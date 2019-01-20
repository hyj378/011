//Ư¡�� ����->���->¦�� �� �̹��� ã��, �̹��� ��ȯ->����
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
//�̹������� Ư¡�� ã�� �Լ�
std::vector<KeyPoint> findkeypoint(Mat img)
{
	//�̹������� Ư¡�� ã�� (SIFT��)
	std::vector<KeyPoint> arrayKey;
	double nMinHessian = 400.;
	Ptr<xfeatures2d::SiftFeatureDetector> Detector = xfeatures2d::SiftFeatureDetector::create(nMinHessian);

	Detector->detect(img, arrayKey);
	return arrayKey;
}

int main()
{
	//int num = 0;
	//std::cout << "�Է��� ������ ������?(3�Է�)" << std::endl;
	//std::cin >> num; //���߿� string���� �ּҵ� �Է¹޴°ɷ�

	Mat CMatLimg;
	Mat CMatMimg;
				 //��������??�Ƹ� �׳� ������ ���ϰ� �ϱ�����...
	std::vector<Mat> Matoriginal_img;
	Matoriginal_img.push_back(CMatLimg);
	Matoriginal_img.push_back(CMatMimg);
	Matoriginal_img.push_back(CMatRimg);

	//�̹��� �ε�	//L�ϰ� M ��Ī
	CMatLimg = imread("C:/Users/Ȳ����/Pictures/A.jpg", IMREAD_COLOR);
	CMatMimg = imread("C:/Users/Ȳ����/Pictures/B.jpg", IMREAD_COLOR);

	if (CMatLimg.empty() || CMatMimg.empty())
	{
		std::cout << "image load fail" << std::endl;
		return -1;
	}

	//���� ������ ���� grayscale�� �ٲٰ� ������ ũ�⸦ ������
	Mat MatLimg;
	Mat MatMimg;

	//ũ��� �׳� ���� ������ ������...
	Size size(CMatLimg.cols / 2, CMatLimg.rows / 2);
	resize(CMatLimg, CMatLimg, size);
	resize(CMatMimg, CMatMimg, size);

	cvtColor(CMatLimg, MatLimg, CV_RGB2GRAY);
	cvtColor(CMatMimg, MatMimg, CV_RGB2GRAY);


	//�̹������� Ư¡�� ã�� (SIFT��)
	std::vector<KeyPoint> LimgKey, MimgKey;
	double nMinHessian = 400.;
	Ptr<xfeatures2d::SiftFeatureDetector> Detector = xfeatures2d::SiftFeatureDetector::create(nMinHessian);

	std::vector<std::string> matImgarray;
	matImgarray.push_back("MatLimg");
	matImgarray.push_back("MatMimg");

	LimgKey = findkeypoint(MatLimg);//Ű����Ʈ������ ��鿵�󿡼� ����
	MimgKey = findkeypoint(MatMimg);

	//Ư¡�� ����ϱ�
	Ptr<xfeatures2d::SiftDescriptorExtractor> Extractor = xfeatures2d::SiftDescriptorExtractor::create();
	Mat matkeypointJaeyeon, matkeypointYuyeon, matkeypointBoth;

	Extractor->compute(CMatLimg, LimgKey, matkeypointJaeyeon);
	Extractor->compute(CMatMimg, MimgKey, matkeypointBoth);


	//Ư¡�� ��Ī�ؼ� ���� ū ��Ī vector ��ȯ
	//Ư¡�� ��Ī�ϱ� (FLANN)
	FlannBasedMatcher Matcher;
	std::vector<DMatch> matches;
	Matcher.match(matkeypointJaeyeon, matkeypointBoth, matches);

	//match list���� �Ӱ谪�� �̿��� good match�� ���
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

	std::vector<DMatch> good_matches;
	for (int i = 0; i < matches.size(); i++)
	{
		if (matches[i].distance < dMinDist * 5)
		{
			good_matches.push_back(matches[i]);
		}
	}
	Mat matGoodMatches;
	drawMatches(CMatLimg, LimgKey, CMatMimg, MimgKey, good_matches, matGoodMatches, Scalar::all(-1), Scalar(-1), std::vector<char>(), DrawMatchesFlags::DEFAULT);
	imshow("allmatches", matGoodMatches);
	waitKey(0);

	//���� �ŷڵ� ���� ��Ī���� �̿��Ͽ� ��ȯ����� (ȣ��׷��Ƿ� ã��)
	std::vector<Point2f> pointLimg;
	std::vector<Point2f> pointMimg;

	for (int i = 0; i < good_matches.size(); i++)
	{
		pointLimg.push_back(LimgKey[good_matches[i].queryIdx].pt);
		pointMimg.push_back(MimgKey[good_matches[i].trainIdx].pt);
	}

	Mat HomoMatrix = findHomography(pointMimg, pointLimg, CV_RANSAC);

	std::cout << HomoMatrix << std::endl;

	//homoMatrix�� �̿��Ͽ� �̹����� warp
	Mat matResult;

	warpPerspective(CMatMimg, matResult, HomoMatrix, Size(CMatMimg.cols * 2, CMatMimg.rows), INTER_CUBIC);

	Mat matPanorama;

	matPanorama = matResult.clone();

	imshow("warp", matResult);

	Mat matROI(matPanorama, Rect(0, 0, CMatLimg.cols, CMatLimg.rows));
	CMatLimg.copyTo(matROI);

	imshow("panorama", matPanorama);
	std::vector<int> compression_params;

	waitKey(0);

	return 0;
}