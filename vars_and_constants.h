#include "cuda_fft.h"
#include <GL\glew.h>
#include <GL\GLAux.h>
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
#include <Magick++.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

using namespace glm;
using namespace Magick;

const int max_resolution = 512;
float* pos;
const float DIST = 1.f;
const float MAXH = 30.f;
const float SUPP = 1.f;
const float SLOW = -1.f;
int resolution = 50;
GLuint prg; 
Camera c_main;
Camera c_sec;
float lx = 10.f;
float lz = 10.f;
float A_norm = 0.015f;
bool geometry = false;
bool pause = false;
vec2 wind = vec2(2.f, 3.f);
float g = 9.81f;
cudaGraphicsResource * resource1;
cudaGraphicsResource * resource2;
cudaGraphicsResource * resource3;
cudaGraphicsResource * resource4;
GLuint density;
GLuint h_field;
GLuint normal_x_field;
GLuint normal_y_field;
GLuint tex;
GLuint density_tex;
GLuint buf_tex;
GLuint buf_index;
GLuint tex_sky;
vec3 sun_direction = vec3(0.89, -0.27, 0.43);
float inner_level = 1.f;
float outer_level = 1.f;
std::vector <int> index;
std::string nameSaved = "";
std::string path = "C:\\Users\\Asus\\Documents\\Cameras\\";
vec4 c0 = vec4(0, 0.13, 0.25, 0.3);
vec4 c90 = vec4(0, 0.27, 0.39, 0.3);
//vec3 sky = vec3(0.61, 0.46, 0.80);
vec3 specular = vec3(0.81, 0.7, 0.23);
float specular_strength = 20.f;
float specular_power = 100;
UINT TextureArray[1];	
float koef_density = 5;
float inner_big_part = 1;
float koef_inner_density = 0.2;
float koef_outter_density = 1;
float outer_big_part = 1;
float number_texel = 10;

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