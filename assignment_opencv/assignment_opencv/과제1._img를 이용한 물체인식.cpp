//�̹��� ���ؼ� ���ϴ� ��� ã��

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
	//���� -> ��� -> ��Ī
	Mat img1, img2;
	img1 = imread("C:/opencv_source/image/model3.jpg", IMREAD_GRAYSCALE);
	img2 = imread("C:/opencv_source/image/scene.jpg", IMREAD_GRAYSCALE);
	if (!(img1.data && img2.data))
	{
		printf("no img");
		return 0;
	}
	//*****************1.���� with sift (SiftFeatureDetector)
	/*shif Ŭ�������� create �Լ��� �ִ�
	create �Լ��� sift�˰����� ���̾��, �Ӱ谪, �ñ׸���(��뿡 ����� ���콺�л갪), ������ Ư¡�� ���� ��ȯ�Ѵ�
	shif Ŭ������ sift�˰����� �̿��� Ű����Ʈ�� �����ϰ� ����ڸ� ã�Ƴ���*/
	Ptr<xfeatures2d::SIFT> instance_FeatureDetector = xfeatures2d::SIFT::create();//������ ���� �ν��Ͻ� ����
																				  //virtual void cv::Feature2D::detect() �̹��� �Ǵ� �̹��� ��Ʈ�� Ű����Ʈ�� �����մϴ�.
	std::vector<KeyPoint> img1keypoint, img2keypoint;

	instance_FeatureDetector->detect(img1, img1keypoint);//Feature2D�� ��ӹ޾����Ƿ� ��밡��
	instance_FeatureDetector->detect(img2, img2keypoint);

	//ã�� keypoint�� �ð������� ǥ��
	//2D Features Framework/modules,Drawing Function of Keypoints and Matches

	Mat displayOfImg1;
	drawKeypoints(img1, img1keypoint, displayOfImg1, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);//������ DRAW_RICH_KEYPOINTS�� ��Ÿ��
	namedWindow("img1�� Ű����Ʈ"); imshow("img1�� Ű����Ʈ", displayOfImg1);

	Mat displayOfImg2;
	drawKeypoints(img2, img2keypoint, displayOfImg2, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);//������ DRAW_RICH_KEYPOINTS�� ��Ÿ��
	namedWindow("img2�� Ű����Ʈ"); imshow("img2�� Ű����Ʈ", displayOfImg2);
	//Ű����Ʈ ���� ������ Ȯ�������Ƿ� Ű����Ʈ ����ϱ�


	//*****************1.��� with sift (SiftDescriptorExtractor)
	Ptr<xfeatures2d::SIFT> instance_Descriptor = xfeatures2d::SIFT::create();//����� ���� �ν��Ͻ� ����
																			 //�� �������� ����ؾ��� ����� ���͸� output...
																			 //2D �̹������� Ư¡�� �����ϰ� ����ϴ� �߻����� �⺻ class Feature2D�� �׸� ��ӹ��� xfeatures2d
																			 //Feature2D�� ����� ã�� �Լ��� compute�ϰ�  detectAndCompute�ۿ� ����.
																			 //���� void??  virtual void�� ����??
																			 //�� �Լ��� ���� ���ǵǾ� �����ʰ�, ������ ����ϴ� Ŭ�������� ������ ���̴� ��� ��������� �˷��ִ� ���̴�. 
																			 //c++�� ���ø� ���������ǰ�??
	Mat img1outputarray, img2outputarray;
	instance_Descriptor->compute(img1, img1keypoint, img1outputarray);
	instance_Descriptor->compute(img2, img2keypoint, img2outputarray);


	//*****************1.��Ī with sift (SiftDescriptorExtractor)
	//�켱 ���� ��Ī�� ����  FLANN�� ������� (BF�� �Ĳ��ؼ� ������ �����ɸ��ϱ�)
	//��Ī�� ���󿡼� ������ Ư¡���� ���� ����� ���͸� �Ÿ��� �̿��� ��Ī�ϴ°�
	//��Ī����� ������ �� �ִ� �������� �ʿ�
	//���� �͵��� �迭�̿��� ������ mat���̿���.
	//��Ī�� ����� matcher�� ���??
	//Dmatch�� ��Ī ����� ǥ���ϴ� Ŭ������ ���ĸ������� ���ȴ�.

	FlannBasedMatcher FLANNmatcher;
	std::vector<DMatch> match;
	FLANNmatcher.match(img2outputarray, img1outputarray, match); //�� Ű����Ʈ�� ��ġ�� ����Ʈ�� Dmatch(name = match)�� ����

																 //match���� �����͸� � ���� ->knnmatch�� ������
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
	//����ũ �μ��� empty�̹Ƿ�(std::vector< char >()) �̹��� �� �������� ��Ī�� �׸���
	namedWindow("��Ī ���"); imshow("��Ī���", finalOutputImg);


	waitKey();
	return 0;
}