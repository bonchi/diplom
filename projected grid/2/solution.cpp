#include <GL\glew.h>
#include <vector>
#include <gl\freeglut.h>
#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
#include <glm\glm.hpp>
//#include <Magick++.h> 
#include <glm\gtc\matrix_transform.hpp> 
#include <glm/gtc/matrix_access.hpp>
#include <glm\gtc\type_ptr.hpp>
#define PI M_PI

using namespace glm;

GLuint prg; 
 
class Camera {
	vec3 pos_;
 
    float heading_;
    float pitch_;
    float roll_;
 
    float fovy_;
    float aspect_;
    float speed_;


    mat4 view() {
        return  rotate(mat4(1), roll_, vec3(0, 0, 1)) *
                rotate(mat4(1), pitch_, vec3(1, 0, 0)) *
                rotate(mat4(1), heading_, vec3(0, 1, 0)) *
                lookAt(pos_, pos_ + vec3(1, 0, 0), vec3(0, 0, 1));
	}
    
public:
	Camera() : pos_(-5, 0, 0), heading_(), pitch_(), speed_(0.1f)
        , roll_(), fovy_(120.f), aspect_(1.f)
    {}
 
    mat4 mvp() {
        return perspective(fovy_, aspect_, .1f, 1000.f) * view();
    }

	mat4 perm() {
		return perspective(fovy_, aspect_, .1f, 1000.f);	
	}
 
    vec3 dir() {
        return -vec3(column(inverse(view()), 2));
    }
 
    vec3 pos() {
        return pos_;
    }
   
    void heading(float diff) {
        heading_ = fmod((float)heading_ + diff, (float)360.);
    }
 
    void pitch(float diff) {
        pitch_ = max(-90.f, min(90.f, pitch_ + diff));
    }
 
    void roll(float diff) {
        roll_ = fmod((float)roll_ + diff, (float)360.);
    }
 
    void move(float diff) {
        pos_ -= diff * vec3(column(inverse(view()), 2));
    }
 
    void move_side(float diff) {
        pos_ += diff * vec3(column(inverse(view()), 0));
    }
 
    void move_vert(float diff) {
        pos_ += diff * vec3(column(inverse(view()), 1));
    }
 
    void set_aspect(float aspect) {
        aspect_ = aspect;
    }
};

Camera c;
//GLuint m_textureObj;
const int width = 30;
const int height = 30;
const float DIST = 1.f;
const float MAXH = 20.f;
float tex[4 * width * (height - 1)];

void display() {
	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(prg);
	
	GLuint buf_tex;

	glGenBuffers(1, &buf_tex);
	glNamedBufferDataEXT(buf_tex, 4 * width * (height - 1) * sizeof(float), tex, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, buf_tex);
    glVertexAttribPointer(1, 2, GL_FLOAT, false, 0, NULL);
	glEnableVertexAttribArray(1);	
	glDrawArrays(GL_LINE_STRIP, 0, width * 2 * (height - 1));
	mat4 m = c.mvp();
	glUniformMatrix4fv(glGetUniformLocation(prg, "m_mvp"), 1, false, value_ptr(m));
	
	vec3 dir = c.dir();
	vec3 point;
	dir = normalize(dir);
	if (dir.z == 0 || c.pos().z == 0) {
		point = c.pos() + dir * DIST;
		point.z = 0;
	} else {
		float t = - c.pos().z / dir.z;
		if (t < 0) {
			point = c.pos() - t * (c.pos() + dir);
		}
		if (t > 0) {
			point = c.pos() + t * (c.pos() - dir);
		}
	}
	
	float x = c.pos().x;
	float y = c.pos().y;
	if (c.pos().z > MAXH) {
		x /= (c.pos().z / MAXH);
		y /= (c.pos().z / MAXH);
	}
	float z = sqrt(3 * ((x - point.x) * (x - point.x) + (y - point.y) * (y - point.y)));
	vec3 pj_pos = vec3(x, y, z);
	
	mat4 m_cam_per = c.perm();
	mat4 m_pview = lookAt(pj_pos, point, vec3(0, 0, 1));
	mat4 m_proj = inverse(m_cam_per * m_pview);
	glUniformMatrix4fv(glGetUniformLocation(prg, "m_proj"), 1, false, value_ptr(m_proj));

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
	GLuint fshader = glCreateShader(GL_FRAGMENT_SHADER);
	GLuint vshader = glCreateShader(GL_VERTEX_SHADER);
	//GLuint gshader = glCreateShader(GL_GEOMETRY_SHADER);
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
	/*{
		std::ifstream stin("gshader.glsl");
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
		glShaderSource(gshader, 1, (const GLchar **)&buffer, &size);
	}
*/	
	glCompileShader(vshader);
	GLint param;

	glCompileShader(fshader);
	//glCompileShader(gshader);

	prg = glCreateProgram();
	glAttachShader(prg, vshader);
	glAttachShader(prg, fshader);
	//glAttachShader(prg, gshader);
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


int mousex;
int mousey;
bool flag = false;

void motionMouse(int x, int y) {
	if (!flag) return;
	float term = 0.1f;
	c.pitch((y - mousey) * term);
	c.heading((x - mousex) * term);
	mousex = x;
	mousey = y;
	glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON) {
		if (state == GLUT_UP) {
			flag = false;
		} else if (state == GLUT_DOWN) {
			mousex = x;
			mousey = y;
			flag = true;
		}
	}
}

void key(unsigned char k, int x, int y) {
	float term = 0.1f;
	if (k == 'Q') {
		exit(0);
	} 


	if (k == 'q') {
		c.move(term);
	}

	if (k == 'e') {
		 c.move(-term);
	 }

	 if (k == 'd') {
		 c.move_side(term);
	}

	 if (k == 'a') {
		 c.move_side(-term);
	 }

	 if (k == 'w') {
		 c.move_vert(term);
	 }

	 if (k == 's') {
		 c.move_vert(-term);
	 }
	 display();
}

int main(int argc, char ** argv)
{
	/*InitializeMagick(*argv);
	{
		Image master("C:\\Program Files (x86)\\ImageMagick-6.8.4-Q16\\images\\affine.png");
		Image second = master;
		second.resize("640x480");
		Image third = master;
		third.resize("800x600");
		second.write("horse640x480.jpg");
		third.write("horse800x600.jpg");
		return 0;
	}*/
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