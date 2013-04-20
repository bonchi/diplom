#version 330

layout (location = 1) in vec2 texPos;

uniform mat4 m_mvp;
uniform mat4 m_proj;
//uniform vec4 trap[4];


void main() {
	/*vec4 v0 = mix(trap[0], trap[1], texPos.x);
	vec4 v1 = mix(trap[3], trap[2], texPos.x);
	vec4 v = mix(v0, v1, texPos.y);
	v /= v.w;
	gl_Position = m_mvp * v;*/
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
