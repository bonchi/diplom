#include <GL\glew.h>
#include <gl\freeglut.h>
#include <iostream>
#include <string>
#include <fstream>
#include <math.h>
#include <glm\glm.hpp>
#include <Magick++.h> 
#include <glm\gtc\matrix_transform.hpp> 
#include <glm/gtc/matrix_access.hpp>
#include <glm\gtc\type_ptr.hpp>
#define PI 3.14159265

using namespace Magick;
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

void display() {
	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(prg);

	float tex[] = {1, 0, 0, 0, 1, 1, 0, 1};
	float pos[] = {1, -1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0};

	GLuint vao;
	GLuint buf_tex;
	GLuint buf_pos;

	glGenBuffers(1, &buf_pos);
	glNamedBufferDataEXT(buf_pos, sizeof(pos), pos, GL_STATIC_DRAW);

	glGenBuffers(1, &buf_tex);
	glNamedBufferDataEXT(buf_tex, sizeof(tex), tex, GL_STATIC_DRAW);

	glGenVertexArrays(1, &vao);

	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, buf_pos);
	glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, NULL);
	glBindBuffer(GL_ARRAY_BUFFER, buf_tex);
    glVertexAttribPointer(1, 2, GL_FLOAT, false, 0, NULL);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	mat4 m = c.mvp();
	float sun_c[] = {0.5f, 0.05f, 0.2f};
	static const vec3 sun_direction = normalize(vec3(0.2, -0.3, -1));
	float camera_pos [3];
	vec3 cam = c.pos();
	camera_pos[0] = cam.x;
	camera_pos[0] = cam.y;
	camera_pos[0] = cam.z;

	glUniformMatrix4fv(glGetUniformLocation(prg, "m_mvp"), 1, false, value_ptr(m));
	glUniform3fv(glGetUniformLocation(prg, "sun_c"), 1, sun_c);
	glUniform3fv(glGetUniformLocation(prg, "sun_direction"), 1, value_ptr(sun_direction));
	glUniform3fv(glGetUniformLocation(prg, "camera_pos"), 1, camera_pos);

	glFlush();
	glutSwapBuffers();
}

void init()
{
	
	//Image duf;
	//duf.read("DIFFUSE.jpg" );
	//duf.modifyImage();
	//duf.type(TrueColorType);
	//PixelPacket *pix = duf.getPixels(rows,columns,duf.columns(),duf.rows()); 

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

int main(int argc, char * argv[])
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