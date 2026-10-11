# Lab Report — Lab 01: Efficient Separable Gaussian Convolution

*Fill in each section below. Be specific — name the actual methods/parameters you used, since
your report is graded alongside your code.*

## Overview

_1 sentence: What was the goal of this lab?_
The goal of this lab was to implement an efficient Gaussian blur by replacing a direct 2D convolution, which requires \(n^2\) kernel computations, with two separable 1D convolution passes requiring only \(2n\).

## Implementation

_1-2 sentences: Briefly describe the steps you performed to implment efficient convolution._
In `gauss_kernel()`, I generated a normalized float64 Gaussian column vector with length `int(6 * sigma + 1)`. In `my_conv_method()`, I converted the image to float64 and used OpenCV `cv2.sepFilter2D` to perform the two 1D passes with `cv2.BORDER_CONSTANT` zero padding, then converted the clipped result to `uint8`.

## Results

_1-2 sentences: How well did your method perform vs the baseline in the pytests? ._
The implementation passed all 10 pytest checks, including the output shape, `uint8` dtype, kernel shape, and every performance tier. On the test run, `cv2.sepFilter2D` took 0.0399 s compared with 5.1685 s for the naive full 2D `scipy.signal.convolve2d` baseline, or approximately 0.01× the baseline time.
