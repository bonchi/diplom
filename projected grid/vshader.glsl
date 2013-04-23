#version 330

layout (location = 1) in vec2 texPos;

uniform mat4 m_mvp;
uniform mat4 m_proj;
uniform vec4 trap[4];

void main() {
	vec4 Vertex0 = mix(trap[0], trap[1], texPos.x);
	vec4 Vertex1 = mix(trap[3], trap[2], texPos.x);
	vec4 Vertex = mix(Vertex0, Vertex1, texPos.y);
	Vertex /= Vertex.w;
	gl_Position = m_mvp * Vertex;
}