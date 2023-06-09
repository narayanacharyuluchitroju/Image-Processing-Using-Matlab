% Assignment 2 (Image Processing) U01079781
% Note : few functions are defined to reduce the redundacy of the code
% (kmedian_filter, kaverage_filter, guassian_filter and varCal)
% kmedian_filter - to apply median filter on an image
% kaverage_filter - to apply average filter on an image
% guassian_filter - to apply guassian filter on an image
% varCal - to find variance of read, green and blue channels in a rgb image.

clc;
clear all;
%%-------------------------------------------------------------------------
%%-------------------------------------------------------------------------
%% Question 1 A
%% histogram Equalization
I = imread('Lena512.bmp');
I2 = histeq(I);

figure(1)
subplot(2,2,1);
imshow(I)
title('Origial Image');
subplot(2,2,2);
imshow(I2);
title('Histogram Equalized Image');
subplot(2,2,3);
histogram(I);
title('Histogram - Original Image');
subplot(2,2,4);
histogram(I2);
title('Histogram  - Equalized Image');


%% Question 1 B

%% Analyze the Noise
% Read the color image
rgbImage = imread('Lena512.bmp'); % Replace 'your_image.jpg' with the actual image file path

% Split the color image into individual color channels
redChannel = rgbImage(:, :, 1);
greenChannel = rgbImage(:, :, 2);
blueChannel = rgbImage(:, :, 3);

% Calculate the standard deviation of pixel values in each color channel
redStd = std2(redChannel);
greenStd = std2(greenChannel);
blueStd = std2(blueChannel);


figure(2)
subplot(2,3,1)
imshow(redChannel)
title({['Red channel'], ['var: ' num2str(redStd)]})
subplot(2,3,4)
histogram(redChannel)
title('Histogram - Red Channel')

figure(2)
subplot(2,3,2)
imshow(greenChannel)
title({['Green channel'],['var: ' num2str(greenStd)]})
subplot(2,3,5)
histogram(greenChannel)
title('Histogram - Green Channel')

figure(2)
subplot(2,3,3)
imshow(blueChannel)
title({['Blue channel'] ,['var: ' num2str(blueStd)]})
subplot(2,3,6)
histogram(blueChannel)
title('Histogram - Blue Channel')



%   Apply filters

%% Kmedian Filter

kmedian_fimg = kmedian_filter(I);

figure(3)
subplot(3,2,1);
imshow(I)
title('Original Image')
subplot(3,2,3);
histogram(I)
title('Histogram - Original Image')
subplot(3,2,2);
imshow(kmedian_fimg);
title('Medain Filtered Image')
subplot(3,2,4);
histogram(kmedian_fimg)
title('Histogram - Median Filtered Image')
% Create a bar graph to display the variances
[redVar, greenVar, blueVar] = varCal(I);
variances = [redVar, greenVar, blueVar];
channelNames = {'Red', 'Green', 'Blue'};

subplot(3,2,5);
bar(variances);
xticks(1:3);
xticklabels(channelNames);
ylabel('Variance');
title('Variances of RGB Channels (Original)');

% Create a bar graph to display the variances
[redVar, greenVar, blueVar] = varCal(kmedian_fimg);
variances = [redVar, greenVar, blueVar];
channelNames = {'Red', 'Green', 'Blue'};

subplot(3,2,6);
bar(variances);
xticks(1:3);
xticklabels(channelNames);
ylabel('Variance');
title('Variances of RGB Channels(Filtered)');


%% Kaverage Filter
kaverage_fimg = kaverage_filter(I);

% Display the filtered image
figure(4)
subplot(3,2,1);
imshow(I)
title('Original Image')
subplot(3,2,3);
histogram(I)
title('Histogram - Original Image')
subplot(3,2,2);
imshow(kaverage_fimg);
title('Kaverage Filtered Image')
subplot(3,2,4);
histogram(kaverage_fimg)
title('Histogram - Kaverage Filtered Image')
% Create a bar graph to display the variances
[redVar, greenVar, blueVar] = varCal(I);
variances = [redVar, greenVar, blueVar];
channelNames = {'Red', 'Green', 'Blue'};

subplot(3,2,5);
bar(variances);
xticks(1:3);
xticklabels(channelNames);
ylabel('Variance');
title('Variances of RGB Channels(Original)');

% Create a bar graph to display the variances
[redVar, greenVar, blueVar] = varCal(kaverage_fimg);
variances = [redVar, greenVar, blueVar];
channelNames = {'Red', 'Green', 'Blue'};

subplot(3,2,6);
bar(variances);
xticks(1:3);
xticklabels(channelNames);
ylabel('Variance');
title('Variances of RGB Channels(Filtered)');

%% Gaussian filter

guassian_fimg = guassian_filter(I); % function defined at the end

% Display the filtered image
figure(5)
subplot(3,2,1);
imshow(I)
title('Original Image')
subplot(3,2,3);
histogram(I)
title('Histogram - Original Image')
subplot(3,2,2);
imshow(guassian_fimg);
title('Guassian Filtered Image')
subplot(3,2,4);
histogram(guassian_fimg)
title('Histogram - Guassian Filtered Image')

% Create a bar graph to display the variances
[redVar, greenVar, blueVar] = varCal(I);
variances = [redVar, greenVar, blueVar];
channelNames = {'Red', 'Green', 'Blue'};

subplot(3,2,5);
bar(variances);
xticks(1:3);
xticklabels(channelNames);
ylabel('Variance');
title('Variances of RGB Channels(Original)');

% Create a bar graph to display the variances
[redVar, greenVar, blueVar] = varCal(guassian_fimg);
variances = [redVar, greenVar, blueVar];
channelNames = {'Red', 'Green', 'Blue'};

subplot(3,2,6);
bar(variances);
xticks(1:3);
xticklabels(channelNames);
ylabel('Variance');
title('Variances of RGB Channels(Filtered)');


figure(6)
subplot(3,4,1);
imshow(I)
title('Original Image');
subplot(3,4,5);
J=histeq(I)
imshow(J);
title('HistEQ - Original Image');
subplot(3,4,9);
histogram(J);
title('Histogram: HistEQ - Original Image');
subplot(3,4,2);
imshow(kmedian_fimg);
title('Medain Filtered');
subplot(3,4,6);
J=histeq(kmedian_fimg);
imshow(J);
title('HistEQ  - Medain Filtered');
subplot(3,4,10);
histogram(J);
title('Histogram: HistEQ - Medain Filtered');
subplot(3,4,3);
imshow(kaverage_fimg);
title('Average Filtered');
subplot(3,4,7);
J=histeq(kaverage_fimg);
imshow(J)
title('HistEQ  - Average Filtered');
subplot(3,4,11);
histogram(J);
title('Histogram: HistEQ  - Average Filtered');
subplot(3,4,4);
imshow(guassian_fimg);
title('Guassian Filtered');
subplot(3,4,8);
J=histeq(guassian_fimg);
imshow(J)
title('HistEQ  - Guassian Filtered');
subplot(3,4,12);
histogram(J)
title('Histogram: HistEQ  - Guassian Filtered');


%% Observation
% Median Filter -----------------------------------------------------------
% Histogram equalization frequently enhances the contrast of the original Lena image 
% by distributing the pixel intensities over the entire dynamic range. 
% The histogram will thus have a more uniform distribution and a larger range of intensities.
% However, when histogram equalization is used, the height of the bars in the histogram might be slightly reduced. 
% to a photo that has undergone a median filtering process. 
% The median filter reduces intensity variations caused on by noise or minute details in the image and removes outliers.
% The dynamic range has been reduced and the frequency of extreme pixel intensities has decreased thanks to the filtering process.

% Average Filter ----------------------------------------------------------
% The average filter is a smoothing filter that distorts the image by substituting each pixel's value with the average of its surrounding pixels.
% This smoothing procedure typically reduces the variations in pixel intensities between neighbors, resulting in a narrower intensity range.
% When a picture has been filtered with an average filter and histogram equalization is used, 
% Because of the filtering procedure, the intensity distribution in the image has changed, as evidenced by the histogram's shorter bar.

% Guassian Filter----------------------------------------------------------

% The image is slightly blurred by the Gaussian filter, which also reduces noise and high-frequency features.
% Since some pixel intensities may become more comparable as a result of this smoothing effect,
% the histogram may become smaller and have lower peak heights.
% Prior to histogram equalization, applying a Gaussian filter may have the effect of smooothing the image.
% Following the use of a Gaussian filter and histogram equalization, a narrower histogram indicates that 
% The local differences in pixel brightness have been decreased by % the filtering. 
% This might be read as a decrease in the overall sharpness and local contrast of the image, giving the impression that it is smoother or softer.


%% Question 2

%% Question 2 A
% Create an image 4 times the size of the original without affecting the features?

% Resize the image
scale = 4;
I_resized = imresize(I, scale, 'bilinear');

% Display the original and resized images
figure(7);
subplot(1, 2, 1);
imshow(I);
title(['Original Image :' num2str(size(I))]);

subplot(1, 2, 2);
imshow(I_resized);
title(['Resized Image : ' num2str(size(I_resized))]);

% Bilinear interpolation is a technique that is frequently used to resize photographs while maintaining their general characteristics and smoothness.
% It operates by inferring the values of adjacent pixels in the original image and predicting the values of those pixels in the scaled image.
% By taking into account the brightness of many nearby pixels,
% Bilinear interpolation at a percentage aids in preserving the general characteristics and smoothness of the image when scaling. 
% The weights given to the adjacent pixels make sure that the scaled image keeps the key elements. 
% and keeps out obvious distortions or artifacts.
% When scaling up an image using bilinear interpolation, each pixel in the original image contributes to multiple pixels in the resized image. The new pixel values are calculated by considering the intensities of the neighboring pixels in the original image.


%% Question 2 B
%% Affine Transformation
figure(8)
subplot(1,3,1)
imshow(I)
title('Original Image')
% Vertical shear and horizontal stretch:
tform = affine2d([3 0 0; 0.5 1 0; 0 0 1]); 
J = imwarp(I, tform);
subplot(1,3,2)
imshow(J)
title(sprintf('Vertical shear and\nhorizontal stretch'));

% In MATLAB, an affine transformation object is produced using the affine2d function.
% A geometric change called an affine transformation maintains planes, straight lines, and points.
% Transposition, rotation, scaling, shearing, and reflection are a few examples of the operations it can do.

% A vertical shear with a factor of 3 and a horizontal stretch with a factor of 0.5 are represented 
% by the transformation matrix [3 0 0; 0.5 1 0; 0 0 1]. 
% In other words, the image will have its width stretched horizontally and its height shorn vertically.

% Horizontal shear and vertical stretch:
tform = affine2d([1 0.5 0; 0 3 0; 0 0 1]);
J = imwarp(I, tform);
subplot(1,3,3)
imshow(J)
title(sprintf('Horizontal shear and\nvertical stretch'));

% A 0.5-factor horizontal shear and a 3-factor vertical stretch are represented by the transformation matrix [1 0.5 0; 0 3 0; 0 0 1]. 
% The image will be stretched vertically and sheared horizontally as a result.



%% Question 3

%% Question 3 A
%% Image Segmentation

%  K-means clustering on the RGB values of the Lena image and assigns 
% each pixel to one of the specified number of clusters. 
% The resulting segmented image is displayed with each cluster assigned a unique color.

I = imread('Lena512.bmp');
figure(9)
subplot(2,2,1);
imshow(I)
title('Original Image');
subplot(2,2,3);
histogram(I);
title('Hist - Original Image')
[L,Centers] = imsegkmeans(I,2); % numClusters = 2;
B = labeloverlay(I,L);
subplot(2,2,2);
imshow(B)
title('Labeled Image')
subplot(2,2,4);
histogram(B);
title('Hist - Labeled Image')


%% Question 3 B

I = imread('Lena512.bmp');
[L,Centers] = imsegkmeans(I,2); % numClusters = 2;
B = labeloverlay(I,L);
kmedian_sfimg = kmedian_filter(B);
kaverage_sfimg = kaverage_filter(B);
guassin_sfimg = guassian_filter(B);

figure(10)
subplot(2,4,1);
imshow(B);
title('Segmented Image')
subplot(2,4,5);
histogram(B);
title('Hist - Segmented Image')
subplot(2,4,2);
imshow(kmedian_sfimg);
title('Median Filtered Segmented Image')
subplot(2,4,6);
histogram(kmedian_sfimg);
title('Hist - Median Filtered Segmented Image')
subplot(2,4,3);
imshow(B);
title('Average Filtered Segmented Image')
subplot(2,4,7);
histogram(kaverage_sfimg);
title('Hist - Average Filtered Segmented Image')
subplot(2,4,4);
imshow(guassin_sfimg);
title('Guassian Filtered Segmented Image')
subplot(2,4,8);
histogram(guassin_sfimg);
title('Hist - Guassian Filtered Segmented Image')

%% Observation: 
% kmedian ----------------------------------------------------------------- 
% While the histogram for the second segment is unchanged, we can see that the height of the bars in the first segment slightly increases.
% This shows that the first segment's intensity levels were more significantly impacted by the kmedian filter than those in the second segment.
% This may occur if the filter is minimizing the intensity variations within the first segment by smoothing or averaging the pixels in that segment.
% in the second segment either had a fairly uniform distribution of intensity values or that the filter did not appreciably change their values.

% kaverage ----------------------------------------------------------------
% Since the k-average filter is a smoothing filter, it has blurred or averaged part of the image's pixel values.
% The merging effect resulted from the decrease in conspicuous borders between the two parts.
% The rise in the segment 2 histogram's height shows that 
% The pixel values in that section may have slightly shifted because to noise or artifacts created by the filter, 
% leads to a different distribution and an increase in the height of the histogram.

% Guassian ----------------------------------------------------------------
% By lowering high-frequency noise and maintaining the overall structure, this smoothed the image.
% The brightness of adjacent pixels tended to become more comparable as a result of this smoothing effect, 
% which caused the two segments to be combined in the histogram analysis.
% The difference between the two segments has been attenuated or blurred to some extent, according to the reduction in segment heights.

%% Image Noise filter functions ------------------------------------------- 
function kmedian_fimg = kmedian_filter(I)
% Split the color image into individual color channels
redChannel = I(:, :, 1);
greenChannel = I(:, :, 2);
blueChannel = I(:, :, 3);

red_channel_flt = medfilt2(redChannel,[3 3]);
green_channel_flt = medfilt2(greenChannel,[3 3]);
blue_channel_flt = medfilt2(blueChannel,[3 3]);
kmedian_fimg = cat(3,red_channel_flt,green_channel_flt,blue_channel_flt);

end


function kaverage_fimg = kaverage_filter(I)
% Convert the image to double precision for calculations
img = im2double(I);

% Define the size of the filter (k)
k = 3;

% Extract the individual color channels
redChannel = img(:,:,1);
greenChannel = img(:,:,2);
blueChannel = img(:,:,3);

% Apply the k-average filter to each color channel
redFiltered = imfilter(redChannel, ones(k) / k^2);
greenFiltered = imfilter(greenChannel, ones(k) / k^2);
blueFiltered = imfilter(blueChannel, ones(k) / k^2);

% Combine the filtered color channels back into a single color image
kaverage_fimg = cat(3, redFiltered, greenFiltered, blueFiltered);
end


function guassian_fimg = guassian_filter(I)
% Convert the image to double precision for calculations
img = im2double(I);

% Define the standard deviation for the Gaussian filter
sigma = 2;

% Define the size of the filter (the extent of the filter is determined by the sigma)
filterSize = 2 * ceil(3 * sigma) + 1;

% Extract the individual color channels
redChannel = img(:,:,1);
greenChannel = img(:,:,2);
blueChannel = img(:,:,3);

% Apply the Gaussian filter to each color channel
redFiltered = imgaussfilt(redChannel, sigma, 'FilterSize', filterSize);
greenFiltered = imgaussfilt(greenChannel, sigma, 'FilterSize', filterSize);
blueFiltered = imgaussfilt(blueChannel, sigma, 'FilterSize', filterSize);

% Combine the filtered color channels back into a single color image
guassian_fimg = cat(3, redFiltered, greenFiltered, blueFiltered);
end


function [redVar, greenVar, blueVar] = varCal(I)
% Convert the image to double precision for calculations
image = im2double(I);

% Calculate the variance of each color channel
redChannel = image(:, :, 1);
greenChannel = image(:, :, 2);
blueChannel = image(:, :, 3);

redVar = var(redChannel(:));
greenVar = var(greenChannel(:));
blueVar = var(blueChannel(:));
end