//특징점 추출->기술->짝이 될 이미지 찾기, 이미지 변환->블랜딩
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;
//이미지에서 특징점 찾는 함수
vector<KeyPoint> findkeypoint(Mat img)
{
	//이미지에서 특징점 찾기 (SIFT로)
	vector<KeyPoint> arrayKey;
	double nMinHessian = 400.;
	Ptr<xfeatures2d::SiftFeatureDetector> Detector = xfeatures2d::SiftFeatureDetector::create(nMinHessian);

	Detector->detect(img, arrayKey);
	return arrayKey;
}
//또 다른 파노라마 코드
//https://stackoverflow.com/questions/23492878/image-disappear-in-panorama-opencv
Mat deleteBlackZone(const Mat &image)
{
	Mat resultGray;
	Mat result;
	image.copyTo(result);

	cvtColor(image, resultGray, CV_RGB2GRAY);

	medianBlur(resultGray, resultGray, 3);
	// 첫번째 매개변수 : 원본 이미지, 두번째 : 필터를 거친 이미지 세번째 : 작은영역의 사이즈
	Mat resultTh;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	threshold(resultGray, resultTh, 1, 255, 0);
	findContours(resultTh, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	//영상 내에서 연결 컴포넌트의 외곽선을 추출하는 함수
	//CV_RETR_EXTERNAL : 외부 외곽선 검색
	//CV_CHAIN_APPROX_SIMPLE : 마지막 점이 수평 또는 수직, 대각선 외곽선에 포함됨

	Mat besta = Mat(contours[0]);

	Rect a = boundingRect(besta);
	cv::Mat half(result, a);
	return half;

}
//이미지 2개 붙이고 붙은 사진을 return
Mat addtwo(Mat matbody, Mat matarm) 
{
	//imshow("startfrombody", matbody);
	//imshow("startfromarm", matarm);

	//계산 속도를 위해 흑백으로
	Mat matgraybody, matgrayarm;
	cvtColor(matbody, matgraybody, CV_RGB2GRAY);
	cvtColor(matarm, matgrayarm, CV_RGB2GRAY);

	//두 이미지에서 keypoint 검출
	vector<KeyPoint> keyPointbody, keyPointarm;
	keyPointbody = findkeypoint(matgraybody);
	keyPointarm = findkeypoint(matgrayarm);

	//두 이미지에서 키포인트를 기술하기
	Ptr<xfeatures2d::SiftDescriptorExtractor> Extractor = xfeatures2d::SiftDescriptorExtractor::create();
	Mat matExtractorbody, matExtractorarm;
	Extractor->compute(matbody, keyPointbody, matExtractorbody);
	Extractor->compute(matarm, keyPointarm, matExtractorarm);

	//키포인트를 매칭하고 매칭정보 DMatch에 저장 후 good_match만 골라낸다.
	//ransac이 badmatch는 걸러주는거 아닌가?? 왜 goodmatch를 고른다음에 다시하지?
	FlannBasedMatcher Matcher;
	vector<DMatch> matches;
		//cv::DescriptorMatcher::match(쿼리, 트레인)
	Matcher.match(matExtractorbody, matExtractorarm, matches);

	//신뢰할 수 있는 매치 골라내기
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

	//신뢰할 수 있는 매칭쌍 good_match를 이용해 변환행렬을 찾자
	//findhomography는 ransac알고리즘으로 변환행렬을 찾아준다
	//변환행렬의 shape에 대해 궁금... 반환형과 if반환형이 정해져있다면 몇몇 변환api에 지정된
	//mat형 (2*2, 3*3, 4*4)는 뭐지? findhomography는 인수로 point를 받으니 Dmatch를 point로 변환
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
	//ROI영상이란 관심영역이라는 뜻이다.
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
	//std::cout << "입력할 사진의 갯수는?(3입력)" << std::endl;
	//std::cin >> num; //나중에 string으로 주소도 입력받는걸로

	Mat CMatLimg;//재현
	Mat CMatMimg;//중간
	Mat CMatRimg;//유연

	//이미지 로드
	CMatLimg = imread("C:/Users/황유진/Pictures/L.jpg", IMREAD_COLOR);
	CMatMimg = imread("C:/Users/황유진/Pictures/M.jpg", IMREAD_COLOR);
	CMatRimg = imread("C:/Users/황유진/Pictures/R.jpg", IMREAD_COLOR);

	if (CMatLimg.empty() || CMatMimg.empty() || CMatRimg.empty())
	{
		cout << "image load fail" << endl;
		return -1;
	}

	//크기는 그냥 원본 사진도 줄이자...
	Size size(CMatLimg.cols / 2, CMatLimg.rows / 2);
	resize(CMatLimg, CMatLimg, size);
	resize(CMatMimg, CMatMimg, size);
	resize(CMatRimg, CMatRimg, size);

	Mat matpanorama;
	//Mat plus(Mat matbody, Mat matarm) 앞에가 body
	matpanorama = addtwo(CMatLimg, CMatMimg);
	matpanorama=deleteBlackZone(matpanorama);
	matpanorama = addtwo(matpanorama, CMatRimg);

	imshow("final", matpanorama);
	waitKey(0);

	return 0;
}