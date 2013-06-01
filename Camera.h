#pragma once
#include <gl\freeglut.h>
#include <string>
#include <sstream>
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp> 
#include <glm/gtc/matrix_access.hpp>
#include <glm\gtc\type_ptr.hpp>
using namespace glm;
#define PI 3.141592653589793238f

class Camera {
	int mousex;
	int mousey;
	bool flag;	

	vec3 pos_;
 
    float heading_;
    float pitch_;
    float roll_;
 
    float fovy_;
    float aspect_;

	mat4 view();
public:
	Camera(void);
	Camera(std::string const &);
	mat4 mvp();
	mat4 perm();
	vec3 pos();
	vec3 dir();

	float getFovy();
	float getPitch();
	float getRoll();
	float getHeading();
	void pitch(float);
	void heading(float);
	void roll(float);
	void move(float);
	void move_side(float);
	void move_vert(float);
	void set_aspect(float);
	void key(unsigned char);
	void mouse(int, int, int, int);
	void motionMouse(int, int);
	void quaternion_from_axisangle (quat &, vec3, float);
	quat getCameraRotation();
	void quaternion_multiply(quat&, quat, quat);
	std::string printMe();
};

