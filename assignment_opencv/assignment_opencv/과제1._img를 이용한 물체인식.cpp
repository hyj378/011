//이미지 비교해서 원하는 대상 찾기

#include <iostream>
#include <stdio.h>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"


using namespace cv;

int main()
{
	//검출 -> 기술 -> 매칭
	Mat img1, img2;
	img1 = imread("C:/opencv_source/image/model3.jpg", IMREAD_GRAYSCALE);
	img2 = imread("C:/opencv_source/image/scene.jpg", IMREAD_GRAYSCALE);
	if (!(img1.data && img2.data))
	{
		printf("no img");
		return 0;
	}
	//*****************1.검출 with sift (SiftFeatureDetector)
	/*shif 클래스에는 create 함수가 있다
	create 함수로 sift알고리즘이 레이어수, 임계값, 시그마값(토대에 적용될 가우스분산값), 보유할 특징의 수를 반환한다
	shif 클래스는 sift알고리즘을 이용해 키포인트를 검출하고 기술자를 찾아낸다*/
	Ptr<xfeatures2d::SIFT> instance_FeatureDetector = xfeatures2d::SIFT::create();//검출을 위한 인스턴스 생성
																				  //virtual void cv::Feature2D::detect() 이미지 또는 이미지 세트의 키포인트를 감지합니다.
	std::vector<KeyPoint> img1keypoint, img2keypoint;

	instance_FeatureDetector->detect(img1, img1keypoint);//Feature2D를 상속받았으므로 사용가능
	instance_FeatureDetector->detect(img2, img2keypoint);

	//찾은 keypoint를 시각적으로 표현
	//2D Features Framework/modules,Drawing Function of Keypoints and Matches

	Mat displayOfImg1;
	drawKeypoints(img1, img1keypoint, displayOfImg1, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);//빨간색 DRAW_RICH_KEYPOINTS로 나타냄
	namedWindow("img1의 키포인트"); imshow("img1의 키포인트", displayOfImg1);

	Mat displayOfImg2;
	drawKeypoints(img2, img2keypoint, displayOfImg2, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);//빨간색 DRAW_RICH_KEYPOINTS로 나타냄
	namedWindow("img2의 키포인트"); imshow("img2의 키포인트", displayOfImg2);
	//키포인트 검출 성공을 확인했으므로 키포인트 기술하기


	//*****************1.기술 with sift (SiftDescriptorExtractor)
	Ptr<xfeatures2d::SIFT> instance_Descriptor = xfeatures2d::SIFT::create();//기술을 위한 인스턴스 생성
																			 //각 점에대해 기술해야함 기술은 벡터를 output...
																			 //2D 이미지에서 특징을 검출하고 기술하는 추상적인 기본 class Feature2D와 그를 상속받은 xfeatures2d
																			 //Feature2D에 기술자 찾는 함수는 compute하고  detectAndCompute밖에 없다.
																			 //가상 void??  virtual void가 뭐지??
																			 //이 함수는 현재 정의되어 있지않고, 누군가 상속하는 클래스에서 정의할 것이다 라고 명시적으로 알려주는 것이다. 
																			 //c++에 템플릿 문법같은건가??
	Mat img1outputarray, img2outputarray;
	instance_Descriptor->compute(img1, img1keypoint, img1outputarray);
	instance_Descriptor->compute(img2, img2keypoint, img2outputarray);


	//*****************1.매칭 with sift (SiftDescriptorExtractor)
	//우선 빠른 매칭을 위해  FLANN을 사용하자 (BF는 꼼꼼해서 연산이 오래걸리니까)
	//매칭은 영상에서 추출한 특징점에 대해 기술한 벡터를 거리를 이용해 매칭하는것
	//매칭결과를 저장할 수 있는 저장형이 필요
	//앞의 것들은 배열이였기 때문에 mat형이였다.
	//매칭의 결과는 matcher에 등록??
	//Dmatch는 매칭 목록을 표현하는 클레스로 정렬목적으로 사용된다.

	FlannBasedMatcher FLANNmatcher;
	std::vector<DMatch> match;
	FLANNmatcher.match(img2outputarray, img1outputarray, match); //두 키포인트를 매치해 리스트를 Dmatch(name = match)에 저장

																 //match에서 좋은것만 몇개 고른다 ->knnmatch가 나을까
	double maxd = 0; double mind = match[0].distance;
	for (int i = 0; i < img1outputarray.rows; i++)
	{
		double dist = match[i].distance;
		if (dist < mind) mind = dist;
		if (dist > maxd) maxd = dist;
	}

	std::vector<DMatch> good_match;
	for (int i = 0; i < img1outputarray.rows; i++)
		if (match[i].distance <= max(2 * mind, 0.02)) good_match.push_back(match[i]);

	Mat finalOutputImg;
	drawMatches(img1, img1keypoint, img2, img2keypoint, good_match, finalOutputImg, Scalar(150, 30, 200), Scalar(0, 0, 255), std::vector< char >(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//마스크 인수가 empty이므로(std::vector< char >()) 이미지 전 영역에서 매칭을 그린다
	namedWindow("매칭 결과"); imshow("매칭결과", finalOutputImg);


	waitKey();
	return 0;
}