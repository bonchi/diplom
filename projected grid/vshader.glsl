#version 330

#define MAX_WAVES_RESOLUTION 17
#define PI 3.1415f

layout (location = 1) in vec3 texPos;

uniform mat4 m_mvp;
uniform int waves_resolution;
uniform float lx;
uniform float lz;
uniform vec2 h_koff[MAX_WAVES_RESOLUTION * MAX_WAVES_RESOLUTION];
uniform vec4 trap[4];

void main() {
	vec4 Vertex0 = mix(trap[0], trap[1], texPos.x);
	vec4 Vertex1 = mix(trap[3], trap[2], texPos.x);
	vec4 Vertex = mix(Vertex0, Vertex1, texPos.y);
	Vertex /= Vertex.w;
	float term = 0;
	for (int i = - waves_resolution / 2; i < waves_resolution / 2 - 1; ++i) {
		for (int j = - waves_resolution / 2; j < waves_resolution / 2 - 1; ++j) {
			vec2 koff = h_koff[(waves_resolution + 1) * (i + waves_resolution / 2) + (j + waves_resolution / 2)];
			vec2 k = vec2(2 * PI * i / lx,2 * PI * j / lz);
			float t = dot(Vertex.xy, k);
			term += koff.x * cos(t)  - koff.y * sin(t);
		}
	}
	Vertex.z = term;
	gl_Position = m_mvp * Vertex;
}