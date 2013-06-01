#version 400 core
#define MAX_WAVE_RESOLUTION 32

layout (location = 1) in vec3 texPos;

uniform mat4 m_mvp;
uniform float lx;
uniform float lz;
uniform vec4 trap[4];
uniform sampler2D tex_tex;

void main() {
	vec4 Vertex0 = mix(trap[0], trap[1], texPos.x);
	vec4 Vertex1 = mix(trap[3], trap[2], texPos.x);
	vec4 Vertex = mix(Vertex0, Vertex1, texPos.y);
	Vertex /= Vertex.w;
	gl_Position = vec4(Vertex);
}