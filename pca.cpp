#include <opencv2/opencv.hpp>
#include <fstream>
using namespace cv;

#define HUMAN_COUNT 51
#define PIC_COUNT	5
#define PIXEL_COUNT PIC_WIDTH*PIC_HEIGHT

#define PIC_HEIGHT 100
#define PIC_WIDTH 100

#define RATE 0.9

std::ofstream file_out; 
bool Compare(std::pair<float,int> a,std::pair<float,int> b)
{
	return a.first < b.first;
}
Mat resize_mat(const Mat temp); 

int MPCA(Mat& eigen_proj,Mat& src_proj,Mat& mean_mat);

float calculate_match_rate(Mat eigen_proj,Mat src_proj,Mat mean_mat,int eigen_value_count);

void main()
{
	file_out = std::ofstream("output.txt");
	int i,j;

	Mat eigen_vector , src_eigen_values , mean_mat;
	int eigen_value_count = MPCA(eigen_vector,src_eigen_values,mean_mat);
	//
	// read a test image not included in study samples
	//
	Mat read_face = imread("ORL/s3/9.bmp",CV_LOAD_IMAGE_GRAYSCALE);

	Mat project_image(read_face.cols*read_face.rows,1,CV_32F);
	
	project_image = resize_mat(read_face);

	project_image = project_image - mean_mat;

	// R * 1
	project_image = eigen_vector * project_image ;

	//std::cout<<"project_image rows "<<project_image.rows<<std::endl;

	float match_rate =  calculate_match_rate(eigen_vector,src_eigen_values,mean_mat,eigen_value_count);
	std::cout<<"match rate :"<<match_rate<<std::endl;

	//
	// calculate the Euclid Distance between project_image and every project_eigenvector
	// find the smallest distance and the id
	//
	
	vector<std::pair<float,int>> result_distance;

	for (i = 0; i<HUMAN_COUNT*PIC_COUNT;i++)
	{
		float tempmin = 0.f;
		for (j = 0; j<eigen_value_count; j++)
		{
			tempmin += (src_eigen_values.at<float>(j,i)- project_image.at<float>(j,0))*(src_eigen_values.at<float>(j,i)- project_image.at<float>(j,0)) ; 
		}
		tempmin = sqrt(tempmin); 
		
		result_distance.push_back(std::pair<float,int>(tempmin,i+1));
	}
	std::sort(result_distance.begin(),result_distance.end(),Compare);

//	std::cout<<mini<<" "<<result_distance.begin()->second/5+1<<std::endl;

	int match_human_number = result_distance.begin()->second/5+1;
	std::cout<<"Human Number\t\t\t:"<<match_human_number<<std::endl;

	Mat result_image(400,1000,CV_8UC3);


	putText(result_image,"Read Face",Point(5,10),CV_FONT_HERSHEY_PLAIN,1.0,Scalar(255,0,0));
	putText(result_image,"Eigen Face",Point(5,140),CV_FONT_HERSHEY_PLAIN,1.0,Scalar(255,0,0));
	putText(result_image,"Match Face",Point(5,275),CV_FONT_HERSHEY_PLAIN,1.0,Scalar(255,0,0));

	Mat result_read_face = result_image(Rect(10,15,PIC_WIDTH,PIC_HEIGHT));
	cvtColor(read_face,result_read_face,CV_GRAY2BGR);
	//
	// show the Number ? eigen face
	//
	int eigen_face_no;
	Mat eigen_face1(PIC_HEIGHT,PIC_WIDTH,CV_8U);
	Mat eigen_face2(PIC_HEIGHT,PIC_WIDTH,CV_32F);
	for (eigen_face_no = 0;eigen_face_no < 10; eigen_face_no++)
	{
		for (i = 0; i< PIC_HEIGHT;i++)
		{
			for (j = 0; j< PIC_WIDTH; j++)
			{
				eigen_face2.at<float>(i,j) = eigen_vector.at<float>(eigen_face_no,i*PIC_WIDTH+j)+mean_mat.at<float>(i*PIC_WIDTH+j,0);
				eigen_face1.at<unsigned char>(i,j) = eigen_vector.at<float>(eigen_face_no,i*PIC_WIDTH+j)+mean_mat.at<float>(i*PIC_WIDTH+j,0);
			}
		}

		Mat result_eigen_face =  result_image(Rect(97*eigen_face_no+10,150,PIC_WIDTH,PIC_HEIGHT));
		cvtColor(eigen_face1,result_eigen_face,CV_GRAY2BGR);
	}

	char match_face_path[255]="";

	for ( i= 0; i< 10 ;i++)
	{
		Mat result_match_face = result_image(Rect(97*i+10,285,PIC_WIDTH,PIC_HEIGHT));

		if (result_distance[i].second%5)
			match_human_number = result_distance[i].second/5+1;
		else 
			match_human_number = result_distance[i].second/5;
		
		sprintf(match_face_path,"ORL/s%d/%d.bmp",match_human_number,result_distance[i].second-(match_human_number-1)*PIC_COUNT);
		Mat match_face = imread(match_face_path,CV_LOAD_IMAGE_GRAYSCALE);
		cvtColor(match_face,result_match_face,CV_GRAY2BGR);
	}

	imshow("result image",result_image);

	waitKey(0);
}

Mat resize_mat(const Mat temp)
{
	Mat re_mat(temp.cols*temp.rows,1,CV_32F);
	int i,j;
	for (i = 0;i< temp.rows; i++)
	{
		for (j = 0;j< temp.cols; j++)
		{
			re_mat.at<float>(i*temp.cols+j) = temp.at<unsigned char>(i,j);
		}
	}
	return re_mat;
}

int MPCA(Mat& eigen_proj,Mat& src_proj,Mat& mean_mat)
{
	Mat src(PIXEL_COUNT,HUMAN_COUNT*PIC_COUNT,CV_32F);

	// load all sample images
	int i,j;
	for (i = 0;i<HUMAN_COUNT; i++)
	{
		for (j = 0;j<PIC_COUNT; j++)
		{
			char path[255]={0};
			sprintf(path,"ORL/s%d/%d.bmp",i+1,j+1);
			Mat temp = imread(path,CV_LOAD_IMAGE_GRAYSCALE);
			int a,b;
			for (a = 0; a<temp.rows; a++)
			{	
				for (b = 0; b<temp.cols; b++)
					src.at<float>(a*temp.cols+b,i*PIC_COUNT+j) = (float)temp.at<unsigned char>(a,b);
			}
		}
	}
	
	// calculate mean vector

	Mat mean(PIXEL_COUNT,1,CV_32F);
	for (j = 0;j<PIXEL_COUNT; j++)
	{
		float temp_total = 0;
		for (i = 0;i<HUMAN_COUNT*PIC_COUNT; i++)
		{
			temp_total += src.at<float>(j,i);
		}
		mean.at<float>(j,0) = (temp_total / float(HUMAN_COUNT*PIC_COUNT));
	}

	Mat mean_face(PIC_HEIGHT,PIC_WIDTH,CV_8U);
	for (i = 0; i< PIC_HEIGHT;i++)
	{
		for (j = 0; j< PIC_WIDTH; j++)
		{
			mean_face.at<unsigned char>(i,j) = mean.at<float>(i*PIC_WIDTH+j,0);
		}
	}
	mean_mat = mean;

	//
	// substract mean vector
	// W = X - Mean
	//
	for (i = 0; i<HUMAN_COUNT*PIC_COUNT; i++)
	{
		for (j = 0; j<PIXEL_COUNT; j++)
		{
			src.at<float>(j,i) = src.at<float>(j,i)-mean.at<float>(j,0);
		}
	}

	//
	// calculate eigenvectors of covariance matrix
	// C = W * W^T ==> W^T * W
	//
	Mat srcT (src.cols,src.rows,src.type());
	transpose(src,srcT);

	Mat mulWTW = srcT*src;

	Mat eigenvalues,eigenvectors;

	eigen(mulWTW,eigenvalues,eigenvectors);

	//
	// calculate sum of all eigenvalues, and accumlate eigenvalues until its rate is larger than RATE(90%) 
	//
	double sum_eigenvalues = sum(eigenvalues)[0];

	// rate 
	double temp_total_value = 0.f;
	for (i = 0;i<eigenvalues.rows;i++)
	{
		temp_total_value += eigenvalues.at<float>(i,0);
		if (temp_total_value/sum_eigenvalues > RATE)
			break;
	}

	int R = i;					// rate of id 

	std::cout<<"Number of large eigenvalues\t:"<<R<<std::endl;

	eigenvalues = eigenvalues.rowRange(Range(0,R));

	eigenvectors = eigenvectors.colRange(Range(0,R));

	//
	// Project eigenvectors to source image matrices
	// U = W * d ( N * R )
	//
	Mat NR,NRT;
	NR = src * eigenvectors;

	transpose(NR,NRT);

	// R * N * N * M = R * M
	Mat Projections = NRT * src;

	file_out<<Projections;

	eigen_proj = NRT;
	src_proj = Projections;

	return R;
}

float calculate_match_rate(Mat eigen_proj,Mat src_proj,Mat mean_mat,int eigen_value_count)
{
	float rate = 0.f;
	float match_count = 0;

	Mat temp_image;
	char image_path[255] ="";

	int m,n,i,j;
	for (m = 1; m<= HUMAN_COUNT;m++)
	{
		for (n = 6;n<= 5+PIC_COUNT;n++)
		{
			sprintf(image_path,"ORL/s%d/%d.bmp",m,n);
			temp_image = imread(image_path,CV_LOAD_IMAGE_GRAYSCALE);

			Mat resize_temp_image = resize_mat(temp_image);

			resize_temp_image = resize_temp_image - mean_mat;

			resize_temp_image = eigen_proj * resize_temp_image;

			double minv = 1000000000000000000;
			int mini = 0;
			for (i = 0; i<PIC_COUNT*HUMAN_COUNT;i++)
			{
				double tempmin = 0.0f;
				for (j = 0;j<eigen_value_count;j++)
				{
					tempmin += (resize_temp_image.at<float>(j,0)-src_proj.at<float>(j,i))*(resize_temp_image.at<float>(j,0)-src_proj.at<float>(j,i));
				}
				tempmin = sqrt(tempmin);
				if (tempmin <minv)
				{
					minv = tempmin;
					mini = i;
				}
			}
			int match_human;
			mini++;
			if (mini%5)
				match_human = (mini)/5+1;
			else 
				match_human = (mini)/5;
			if (match_human == m)
				match_count++;

			std::cout<<"human number :"<<m<<" match number :"<<match_human<<std::endl;
		}
	}
	rate = match_count / (HUMAN_COUNT*PIC_COUNT);

	return rate;
}