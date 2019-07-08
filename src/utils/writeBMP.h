#ifndef __WRITE_BMP__
#define __WRITE_BMP__
#include <stdlib.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
//define pixelformat of windows bitmap, notice the unusual ordering of colors
typedef struct {
	unsigned char B;
	unsigned char G;
	unsigned char R;
} pixel;
void writeBMP(const char* filename, unsigned int w, unsigned int h, float *pixels);
void writeBMPColor(const char* filename, unsigned int w, unsigned int h, float *first, float *second);
//supply an array of pixels[height][width] <- notice that height comes first
void wrtieBMPuc3 (const char* filename, unsigned int w, unsigned int h, unsigned char *pixels);
#endif // !__WRITE_BMP__




