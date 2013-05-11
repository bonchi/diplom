#include "Camera.h"

Camera::Camera() : pos_(-5, 0, 0), heading_(), pitch_()
	, roll_(), fovy_(120.f), aspect_(1.f), flag(false)
{}

mat4 Camera::view() {
	return  rotate(mat4(1), roll_, vec3(0, 0, 1)) *
		rotate(mat4(1), pitch_, vec3(1, 0, 0)) *
		rotate(mat4(1), heading_, vec3(0, 1, 0)) *
		lookAt(pos_, pos_ + vec3(1, 0, 0), vec3(0, 0, 1));
}
 
float Camera::getPitch() {
	return pitch_;
}
float Camera::getFovy() {
	return fovy_;
}

float Camera::getRoll() {
	return roll_;
}
float Camera::getHeading() {
	return heading_;
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
	} else if (button == 3) {
		if (fovy_ < 179) {
			fovy_ += 1.f;
		}
	} else if (button == 4) {
		if (fovy_ > 1) {
			fovy_ -= 1.f;
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
		roll(10 * term);
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

void Camera::quaternion_from_axisangle (quat &q, vec3 v, float a) {
	float sin_a = sin(a * 0.5 * PI / 180);
	float cos_a = cos(a * 0.5 * PI / 180);
	q.x = v.x * sin_a;
	q.y = v.y * sin_a;
	q.z = v.z * sin_a;
	q.w = cos_a;
	q = normalize(q);
}

quat Camera::getCameraRotation() {
	vec3 vx = vec3(1.f, 0, 0);
	vec3 vy = vec3(0, 1.f, 0);
	vec3 vz = vec3(0, 0, 1.f);
	quat q, qx, qy, qz, qterm;
	quaternion_from_axisangle(qx, vx, pitch_ );
	quaternion_from_axisangle(qy, vy, heading_ );
	quaternion_from_axisangle(qz, vz, roll_ );
	q = qx * qy * qz;
	return q;
}

std::string Camera::printMe() {
	std::stringstream os;
	os << mousex  << " " << mousey << " " << flag << " " << pos_.x << " " << pos_.y << " " << pos_.z << " " << heading_ << " "
		<< pitch_ << " " << roll_ << " " << fovy_ << " " << aspect_;
	return os.str();
}

Camera::Camera(std::string  const & s) {
	std::stringstream in(s);
	in >> mousex  >> mousey >> flag >> pos_.x >> pos_.y >> pos_.z >> heading_ >> pitch_ >> roll_ >> fovy_ >> aspect_;
}
