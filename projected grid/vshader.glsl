#version 330
#define MAX_WAVE_RESOLUTION 128

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
	vec2 term;
	float kx = - (MAX_WAVE_RESOLUTION - 1) * lx / MAX_WAVE_RESOLUTION;
	float kz = - (MAX_WAVE_RESOLUTION - 1) * lz / MAX_WAVE_RESOLUTION;
	term.x = Vertex.x - int(Vertex.x / kx) * kx;
	if (term.x > 0) {
		term.x -= kx;
	}
	term.y = Vertex.y - int(Vertex.y / kz) * kz;
	if (term.y > 0) {
		term.y -= kz;
	}
	Vertex.z = texture(tex_tex, vec2(term.x / kx, term.y / kz)).x;
	gl_Position = m_mvp * Vertex;
}