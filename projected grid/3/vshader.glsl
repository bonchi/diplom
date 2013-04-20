#version 330

layout (location = 1) in vec2 texPos;

uniform mat4 m_mvp;	

void main() {
	float x = texPos.x * 2. - 1;
	float y = texPos.y * 2. - 1;
	vec4 near = vec4(x, y, -1.0, 1.0);
	vec4 far = vec4(x, y, 1.0, 1.0);
	mat4 mi = inverse(m_mvp);
	near = mi * near;
	far = mi * far;	
	near /= near.w;
	far /= far.w;
	vec4 term = far - near;
	float k = -near.z / term.z;
	vec3 result = near.xyz + term.xyz * k;
	gl_Position = m_mvp * vec4(result, 1.);
}
