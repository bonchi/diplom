#include "cuda_fft.h"
#include <GL\glew.h>
#include <vector>
#include <gl\freeglut.h>
#include <time.h>
#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
#include <ctime>
#include <complex>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp> 
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "Camera.h"
#include <GL/AntTweakBar.h>
#include <cmath>

using namespace glm;

#define MAX_WAVE_RESOLUTION 128

const int max_resolution = 255;
float* pos;
const float DIST = 1.f;
const float MAXH = 30.f;
const float SUPP = 1.f;
const float SLOW = -1.f;
int resolution = 100;
GLuint prg; 
Camera c_main;
Camera c_sec;
float lx = 10;
float lz = 10.f;
float A_norm = 0.01f;
vec2 wind = vec2(2.f, 3.f);
float g = 9.81f;
float h_koff [2 * MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION];
std::complex<float> h0 [MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION];
std::complex<float> h0_minus [MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION];
float result[MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION];
GLuint tex;
GLuint buf_tex;
GLuint buf_index;

std::vector <int> index;
std::string nameSaved = "";
std::string path = "C:\\Users\\Asus\\Documents\\Cameras\\";

//z
vec4 cube[8] = {
	vec4(1., 1., 1., 1.),
	vec4(1., 1., -1., 1.),
	vec4(1., -1., 1., 1.),
	vec4(1., -1., -1., 1.),
	vec4(-1., -1., 1., 1.),
	vec4(-1., -1., -1., 1.),
	vec4(-1., 1., 1., 1.),
	vec4(-1., 1., -1., 1.)
};
//y
vec4 cube2[8] = {
	vec4(1., 1., 1., 1.),
	vec4(1., -1., 1., 1.),
	vec4(1., 1., -1., 1.),
	vec4(1., -1., -1., 1.),
	vec4(-1., 1., 1., 1.),
	vec4(-1., -1., 1., 1.),
	vec4(-1., 1., -1., 1.),
	vec4(-1., -1., -1., 1.)
};
//x
vec4 cube3[8] = {
	vec4(1., 1., 1., 1.),
	vec4(-1., 1., 1., 1.),
	vec4(1., -1., 1., 1.),
	vec4(-1., -1., 1., 1.),
	vec4(1., -1., -1., 1.),
	vec4(-1., -1., -1., 1.),
	vec4(1., 1., -1., 1.),
	vec4(-1., 1., -1., 1.)
};