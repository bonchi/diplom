#version 330

layout (location = 1) in vec2 texPos;

uniform mat4 m_mvp;
uniform mat4 m_proj;

void main() {
	float x = texPos.x * 2 - 1;
	float y = texPos.y * 2 - 1;
	vec4 near = vec4(x, y, -1.0, 1.0);
	vec4 far = vec4(x, y, 1.0, 1.0);
	near = m_proj * near;
	near /= near.w;
	far = m_proj * far;
	far /= far.w;
	vec4 term = far - near;
	float k = -near.z / term.z;
	vec4 result = near + term * k;
	gl_Position = m_mvp * result;
}