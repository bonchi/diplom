#version 330

layout (location = 0) in vec3 pos;

uniform mat4 m_mvp;

out vec3 worldPos;

void main() {
	worldPos = pos;
	gl_Position = m_mvp * vec4(pos, 1);
}
