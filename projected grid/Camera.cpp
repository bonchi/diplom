#include "Camera.h"

Camera::Camera() : pos_(-5, 0, 0), heading_(), pitch_(), speed_(0.1f)
	, roll_(), fovy_(120.f), aspect_(1.f), flag(false)
{}

mat4 Camera::view() {
	return  rotate(mat4(1), roll_, vec3(0, 0, 1)) *
		rotate(mat4(1), pitch_, vec3(1, 0, 0)) *
		rotate(mat4(1), heading_, vec3(0, 1, 0)) *
		lookAt(pos_, pos_ + vec3(1, 0, 0), vec3(0, 0, 1));
}
 
mat4 Camera::mvp() {
	return perspective(fovy_, aspect_, .1f, 100.f) * view();
}

mat4 Camera::perm() {
	return perspective(fovy_, aspect_, .1f, 100.f);	
}
 
vec3 Camera::dir() {
	return -vec3(column(inverse(view()), 2));
}
 
vec3 Camera::pos() {
	return pos_;
}
   
void Camera::heading(float diff) {
	heading_ = fmod((float)heading_ + diff, (float)360.);
}
 
void Camera::pitch(float diff) {
	pitch_ = max(-90.f, min(90.f, pitch_ + diff));
}
 
void Camera::roll(float diff) {
	roll_ = fmod((float)roll_ + diff, (float)360.);
}
 
void Camera::move(float diff) {
	pos_ -= diff * vec3(column(inverse(view()), 2));
}
 
void Camera::move_side(float diff) {
	pos_ += diff * vec3(column(inverse(view()), 0));
}
 
void Camera::move_vert(float diff) {
	pos_ += diff * vec3(column(inverse(view()), 1));
}
 
void Camera::set_aspect(float aspect) {
	aspect_ = aspect;
}

void Camera::mouse(int button, int state, int x, int y) {
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

void Camera::motionMouse(int x, int y) {
	if (!flag) return;
	float term = 0.1f;
	pitch((y - mousey) * term);
	heading((x - mousex) * term);
	mousex = x;
	mousey = y;
}

void Camera::key(unsigned char k) {
	float term = 0.1f;
	if (k == 'r') {
		roll(term);
	}
	if (k == 'q') {
		move(term);
	}
	if (k == 'e') {
		 move(-term);
	 }
	 if (k == 'd') {
		move_side(term);
	}

	 if (k == 'a') {
		move_side(-term);
	 }

	 if (k == 'w') {
		move_vert(term);
	 }

	 if (k == 's') {
		move_vert(-term);
	 }
}
