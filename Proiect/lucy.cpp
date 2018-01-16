#include <opencv.hpp>
#include <stdio.h>
#include <stdlib.h>


/*

Running command:
	g++ -o lucy lucy.cpp -I /usr/local/Cellar/opencv/3.4.0/include/opencv2 -I /usr/local/Cellar/opencv/3.4.0/include `pkg-config --cflags --libs opencv`

*/


using namespace cv;


/* Lucy-Richardson Deconvolution */
void lucy_richardson_deconv(Mat img, int num_iterations, double psf, Mat& result, int num_steps) {
	// Window size of PSF
	int winSize = 8 * psf + 1, count = 0, interval;

	// calculate number of photos to display
	if (num_steps > 0) {
		interval = num_iterations / num_steps;

		if (num_steps * interval == num_iterations) {
			interval = num_iterations / (num_steps + 1);
		}
	}

	// initializations
	Mat Y = img.clone();
	Mat J1 = img.clone();
	Mat J2 = img.clone();
	Mat wI = img.clone();
	Mat imR = img.clone();
	Mat reBlurred = img.clone();

	Mat T1, T2, tmpMat1, tmpMat2;
	T1 = Mat(img.rows,img.cols, CV_64F, 0.0);
	T2 = Mat(img.rows,img.cols, CV_64F, 0.0);


	// Deconvolution CORE
	double lambda = 0;
	for(int j = 0; j < num_iterations; j++) {
		if (j>1) {
			// calculation of lambda (repair factor)
			multiply(T1, T2, tmpMat1);
			multiply(T2, T2, tmpMat2);
			lambda = sum(tmpMat1)[0] / (sum(tmpMat2)[0]);
		}

		Y = J1 + lambda * (J1-J2);
		Y.setTo(0, Y < 0);

		// 1) Applying Gaussian filter
		GaussianBlur(Y, reBlurred, Size(winSize,winSize), psf, psf);

		// 2) 
		divide(wI, reBlurred, imR);

		// 3) Applying Gaussian filter
		GaussianBlur(imR, imR, Size(winSize,winSize), psf, psf);

		// 4) Restauration of data for next step
		J2 = J1.clone();
		multiply(Y, imR, J1);

		T2 = T1.clone();
		T1 = J1 - Y;

		if (num_steps > 0 && count < num_steps && j % interval == 0 && j != 0) {
				count++;
				std::string title = "Figure ";
				printf("Figure: %d\n", count);

				Mat aux = J1.clone();
				normalize(aux, aux, 0, 1, NORM_MINMAX);

				std::stringstream ss;
				ss << title << count << " (Iteration " << j << ")";

				// display photo
				imshow(ss.str(), aux);
		}
	}

	// output
	result = J1.clone();
}


/* Driver program to test above functions */
int main(int argc, char *argv[]) {
	Mat img, result;
	int mode = -1, num_iterations, num_steps = 0;
	double psf = 6.0;

	if (argc < 2) {
        fprintf(stderr, "Usage: %s <in_file> <num_iterations> (<PSF>) (<num_steps_to_show>) (<mode>)\n", argv[0]);
        exit(1);
    }

    // select mode
	if (argv[5]) {
		// CV_LOAD_IMAGE_UNCHANGED (<0) loads the image as is (including the alpha channel if present)
		// CV_LOAD_IMAGE_GRAYSCALE (0) loads the image as an intensity one
		// CV_LOAD_IMAGE_COLOR (>0) loads the image in the BGR format
		mode = atoi(argv[5]);
	}

	// read image
	img = imread(argv[1], mode);

	// display original image
	imshow("Original", img);

	// convert to double
	img.convertTo(img, CV_64F);

	// number of iterations
	num_iterations = atoi(argv[2]);

	// number of steps to display
	if (argv[4]) {
		num_steps = atoi(argv[4]);
	}

	// point spread function
	if (argv[3]) {
		psf = (double) atof(argv[3]);
	}

	int winSize = 8 * psf + 1 ;

	// Blur the original image
	GaussianBlur(img, img, Size(winSize,winSize), psf, psf);
	normalize(img, img, 0, 1, NORM_MINMAX);

	// display blur image
	imshow("Blur", img);

	// apply Lucy-Richardson Deconvolution
	lucy_richardson_deconv(img, num_iterations, psf, result, num_steps);
	normalize(result, result, 0, 1, NORM_MINMAX);

	// display result
	imshow("Result", result);

	waitKey(0);

	return 0;
}
