#include <GL\glew.h>
#include <vector>
#include <gl\freeglut.h>
#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp> 
#include <glm/gtc/matrix_access.hpp>
#include <glm\gtc\type_ptr.hpp>
#include "Camera.h"
#define PI 3.1415

using namespace glm;

GLuint prg; 
const int width = 30;
const int height = 30;
const float DIST = 1.f;
const float MAXH = 30.f;
const float SUPP = 1.f;
const float SLOW = -1.f;

vec3 buildProjectorPos2(const vec3 &camera_pos) {
	float x = camera_pos.x;
	float y = camera_pos.y;
	float z = camera_pos.z;
	if ((z < SUPP) && (z > SLOW)) {
		if (z < 0) {
			z = SLOW;
		} else {
			z = SUPP;
		}
	}
	return vec3(x, y, z);
}

Camera c;
float tex[4 * width * (height - 1)];
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
GLuint buf_tex;

void display() {
	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(prg);

	mat4 m = c.mvp();

	vec3 point  = c.pos() + normalize(c.dir()) * DIST;
	point.z = 0;
	vec3 cam_dir = c.dir();
	vec3 pj_pos = buildProjectorPos2(c.pos());
	
	mat4 m_pview = lookAt(pj_pos, point, vec3(0, 0, 1));
	mat4 m_proj = inverse(c.perm() * m_pview);

	vec4 rc[8];
	vec4 rc2[8];
	vec4 rc3[8];
	mat4 im = inverse(m);
	for (int i = 0; i < 8; ++i) {
		rc[i] = im * cube[i];
		rc2[i] = im * cube2[i];
		rc3[i] = im * cube3[i];
	}
	vec4 trap[32];
	int count = 0;
	for (int i = 0; i < 4; ++i) {
		//боковые грани
		vec4 term = rc[2 * i] - rc[2 * i + 1];
		if (term.z - term.w * SUPP != 0) {
			float k = (rc[2 * i + 1].w * SUPP - rc[2 * i + 1].z) / (term.z - term.w * SUPP);
			if (k > 0) {
				trap[count] = rc[2 * i + 1] + k * term;
				trap[count].z = 0;
				trap[count] /= trap[count].w;
				++count;
			}
		}
		if (term.z - term.w * SLOW != 0) {
			float k = (rc[2 * i + 1].w * SLOW - rc[2 * i + 1].z) / (term.z - term.w * SLOW);
			if (k >= 0) {
				trap[count] = rc[2 * i + 1] + k * term;
				trap[count].z = 0;
				trap[count] /= trap[count].w;
				++count;
			}
		}
		//точки
		if ((rc[2 * i].z / rc[2 * i].w < SUPP) && (rc[2 * i].z / rc[2 * i].w > SLOW)) {
			trap[count] = rc[2 * i];
			trap[count].z = 0;
			trap[count] /= trap[count].w;
			++count;
		}
		if ((rc[2 * i + 1].z / rc[2 * i + 1].w < SUPP) && (rc[2 * i + 1].z / rc[2 * i + 1].w > SLOW)) {
			trap[count] = rc[2 * i + 1];
			trap[count].z = 0;
			trap[count] /= trap[count].w;
			++count;
		}
		//пересечения для ближней и дальней плоскостей
		term = rc2[2 * i] - rc2[2 * i + 1];
		if (term.z - term.w * SUPP != 0) {
			float k = (rc2[2 * i + 1].w * SUPP - rc2[2 * i + 1].z) / (term.z - term.w * SUPP);
			vec4 p = rc2[2 * i + 1] + k * term;
			p /= p.w;
			if ((rc2[2 * i].z / rc2[2 * i].w - p.z) * (rc2[2 * i + 1].z / rc2[2 * i + 1].w - p.z) <= 0) {
				trap[count] = p;
				trap[count].z = 0;
				trap[count] /= trap[count].w;
				++count;
			}
		}
		if (term.z - term.w * SLOW != 0) {
			float k = (rc2[2 * i + 1].w * SLOW - rc2[2 * i + 1].z) / (term.z - term.w * SLOW);
			vec4 p = rc2[2 * i + 1] + k * term;
			p /= p.w;
			if ((rc2[2 * i].z / rc2[2 * i].w - p.z) * (rc2[2 * i + 1].z / rc2[2 * i + 1].w - p.z) <= 0) {
				trap[count] = p;
				trap[count].z = 0;
				trap[count] /= trap[count].w;
				++count;
			}
		}
		term = rc3[2 * i] - rc3[2 * i + 1];
		if (term.z - term.w * SUPP != 0) {
			float k = (rc3[2 * i + 1].w * SUPP - rc3[2 * i + 1].z) / (term.z - term.w * SUPP);
			vec4 p = rc3[2 * i + 1] + k * term;
			p /= p.w;
			if ((rc3[2 * i].z / rc3[2 * i].w - p.z) * (rc3[2 * i + 1].z / rc3[2 * i + 1].w - p.z) <= 0) {
				trap[count] = p;
				trap[count].z = 0;
				trap[count] /= trap[count].w;
				++count;
			}
		}
		if (term.z - term.w * SUPP != 0) {
			float k = (rc3[2 * i + 1].w * SLOW - rc3[2 * i + 1].z) / (term.z - term.w * SLOW);
			vec4 p = rc3[2 * i + 1] + k * term;
			p /= p.w;
			if ((rc3[2 * i].z / rc3[2 * i].w - p.z) * (rc3[2 * i + 1].z / rc3[2 * i + 1].w - p.z) <= 0) {
				trap[count] = p;
				trap[count].z = 0;
				trap[count] /= trap[count].w;
				++count;
			}
		}
	}	
	
	if (count > 0) {
		mat4 tr = inverse(m_proj);
		for (int i = 0; i < count; ++i) {
			trap[i] = tr * trap[i];
			trap[i] /= trap[i].w;
		}
		float xmin, xmax, ymin, ymax;
		xmin = trap[0].x;
		xmax = trap[0].x;
		ymin = trap[0].y;
		ymax = trap[0].y;
		for (int i = 1; i < count; ++i) {
			if (trap[i].x < xmin) {
				xmin = trap[i].x;
			}
			if (trap[i].x > xmax) {
				xmax = trap[i].x;
			}
			if (trap[i].y < ymin) {
				ymin = trap[i].y;
			}
			if (trap[i].y > ymax) {
				ymax = trap[i].y;
			}
		}

		mat4 m_range = mat4(vec4(xmax - xmin, 0, 0, 0),vec4(0, ymax - ymin, 0, 0), vec4(0, 0, 1, 0), vec4(xmin, ymin, 0, 1));
		mat4 m_proj2 =  m_proj * m_range;

		glBindBuffer(GL_ARRAY_BUFFER, buf_tex);
		glDrawArrays(GL_LINE_STRIP, 0, width * 2 * (height - 1));

		for (int i = 0; i < 8; ++i) {
			rc[i] = m_proj2 * cube[i];
		}
		for (int i = 0; i < 4; ++i) {
			vec4 term = rc[2 * i] - rc[2 * i + 1];
			float k = -rc[2 * i + 1].z / term.z;
			trap[i] = rc[2 * i + 1] + k * term;
		}

		glUniformMatrix4fv(glGetUniformLocation(prg, "m_mvp"), 1, false, value_ptr(m));
		glUniformMatrix4fv(glGetUniformLocation(prg, "m_proj"), 1, false, value_ptr(m_proj2));
		glUniform4fv(glGetUniformLocation(prg, "trap"), 4, value_ptr(trap[0]));
	}
	glFlush();
	glutSwapBuffers();
}

void init() {

	for (int i = 0; i < height - 1; ++i) {
		for (int j = 0; j < width; ++j) {
			tex[4 * (i * width + j)] = ((float) i) / (height - 1);
			tex[4 * (i * width + j) + 1] = ((float) j) / (width - 1);
			tex[4 * (i * width + j) + 2] = ((float) (i + 1)) / (height - 1);
			tex[4 * (i * width + j) + 3] = ((float) j) / (width - 1);	
		}
	}

	glGenBuffers(1, &buf_tex);
	glNamedBufferDataEXT(buf_tex, 4 * width * (height - 1) * sizeof(float), tex, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, buf_tex);
	glVertexAttribPointer(1, 2, GL_FLOAT, false, 0, NULL);
	glEnableVertexAttribArray(1);	
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	GLuint fshader = glCreateShader(GL_FRAGMENT_SHADER);
	GLuint vshader = glCreateShader(GL_VERTEX_SHADER);
	int size;
	{
		std::ifstream stin("vshader.glsl");
		std::string source;
		while (stin)
		{
			std::string line;
			getline(stin, line);
			source += line;
			source += "\n";
		}
		char const * buffer = source.c_str();
		size = source.length();
		glShaderSource(vshader, 1, (const GLchar **)&buffer, &size);
	}
	{
		std::ifstream stin("fshader.glsl");
		std::string source;
		while (stin)
		{
			std::string line;
			getline(stin, line);
			source += line;
			source += "\n";
		}
		char const * buffer = source.c_str();
		size = source.length();
		glShaderSource(fshader, 1, (const GLchar **)&buffer, &size);
	}

	glCompileShader(vshader);
	GLint param;

	glCompileShader(fshader);

	prg = glCreateProgram();
	glAttachShader(prg, vshader);
	glAttachShader(prg, fshader);
	glLinkProgram(prg);

	glGetProgramiv(prg, GL_LINK_STATUS, &param);
	{
		GLint len;
		glGetProgramiv(prg, GL_INFO_LOG_LENGTH, &len);
		char * buff = new char[len];
		glGetProgramInfoLog(prg, len, &len, buff);
		std::cerr << buff << std::endl;
		delete [] buff;
	}
	if (!param)
		throw 1;
}



void motionMouse(int x, int y) {
	c.motionMouse(x, y);
	glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) {
	c.mouse(button, state, x, y);
}

void key(unsigned char k, int x, int y) {
	if (k == 'Q') {
		exit(0);
	} 
	c.key(k);
	display();
}

int main(int argc, char ** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_ACCUM | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(500, 500);

	glutCreateWindow( "window" );

	glutDisplayFunc(display);
	glutKeyboardFunc(key);
	glutMouseFunc(mouse);
	glutMotionFunc(motionMouse);
	glEnable(GL_DEPTH_TEST);
	glewInit();
	init();
	glutMainLoop();

	return 0;
}