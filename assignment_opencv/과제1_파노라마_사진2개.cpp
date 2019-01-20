//특징점 추출->기술->짝이 될 이미지 찾기, 이미지 변환->블랜딩
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
//이미지에서 특징점 찾는 함수
std::vector<KeyPoint> findkeypoint(Mat img)
{
	//이미지에서 특징점 찾기 (SIFT로)
	std::vector<KeyPoint> arrayKey;
	double nMinHessian = 400.;
	Ptr<xfeatures2d::SiftFeatureDetector> Detector = xfeatures2d::SiftFeatureDetector::create(nMinHessian);

	Detector->detect(img, arrayKey);
	return arrayKey;
}

int main()
{
	//int num = 0;
	//std::cout << "입력할 사진의 갯수는?(3입력)" << std::endl;
	//std::cin >> num; //나중에 string으로 주소도 입력받는걸로

	Mat CMatLimg;
	Mat CMatMimg;
				 //얕은복사??아마 그냥 참조를 편하게 하기위해...
	std::vector<Mat> Matoriginal_img;
	Matoriginal_img.push_back(CMatLimg);
	Matoriginal_img.push_back(CMatMimg);
	Matoriginal_img.push_back(CMatRimg);

	//이미지 로드	//L하고 M 매칭
	CMatLimg = imread("C:/Users/황유진/Pictures/A.jpg", IMREAD_COLOR);
	CMatMimg = imread("C:/Users/황유진/Pictures/B.jpg", IMREAD_COLOR);

	if (CMatLimg.empty() || CMatMimg.empty())
	{
		std::cout << "image load fail" << std::endl;
		return -1;
	}

	//빠른 연산을 위해 grayscale로 바꾸고 사진의 크기를 줄이자
	Mat MatLimg;
	Mat MatMimg;

	//크기는 그냥 원본 사진도 줄이자...
	Size size(CMatLimg.cols / 2, CMatLimg.rows / 2);
	resize(CMatLimg, CMatLimg, size);
	resize(CMatMimg, CMatMimg, size);

	cvtColor(CMatLimg, MatLimg, CV_RGB2GRAY);
	cvtColor(CMatMimg, MatMimg, CV_RGB2GRAY);


	//이미지에서 특징점 찾기 (SIFT로)
	std::vector<KeyPoint> LimgKey, MimgKey;
	double nMinHessian = 400.;
	Ptr<xfeatures2d::SiftFeatureDetector> Detector = xfeatures2d::SiftFeatureDetector::create(nMinHessian);

	std::vector<std::string> matImgarray;
	matImgarray.push_back("MatLimg");
	matImgarray.push_back("MatMimg");

	LimgKey = findkeypoint(MatLimg);//키포인트까지만 흑백영상에서 검출
	MimgKey = findkeypoint(MatMimg);

	//특징점 기술하기
	Ptr<xfeatures2d::SiftDescriptorExtractor> Extractor = xfeatures2d::SiftDescriptorExtractor::create();
	Mat matkeypointJaeyeon, matkeypointYuyeon, matkeypointBoth;

	Extractor->compute(CMatLimg, LimgKey, matkeypointJaeyeon);
	Extractor->compute(CMatMimg, MimgKey, matkeypointBoth);


	//특징점 매칭해서 가장 큰 매칭 vector 반환
	//특징점 매칭하기 (FLANN)
	FlannBasedMatcher Matcher;
	std::vector<DMatch> matches;
	Matcher.match(matkeypointJaeyeon, matkeypointBoth, matches);

	//match list에서 임계값을 이용해 good match만 골라냄
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

	//이제 신뢰도 높은 매칭점을 이용하여 변환행렬을 (호모그래피로 찾자)
	std::vector<Point2f> pointLimg;
	std::vector<Point2f> pointMimg;

	for (int i = 0; i < good_matches.size(); i++)
	{
		pointLimg.push_back(LimgKey[good_matches[i].queryIdx].pt);
		pointMimg.push_back(MimgKey[good_matches[i].trainIdx].pt);
	}

	Mat HomoMatrix = findHomography(pointMimg, pointLimg, CV_RANSAC);

	std::cout << HomoMatrix << std::endl;

	//homoMatrix를 이용하여 이미지를 warp
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