#version 330

layout (location = 1) in vec3 texPos;

uniform mat4 m_mvp;
uniform vec4 trap[4];
uniform vec3 waveVector[8];
uniform float ampl[8];
uniform float waveLength[8];
uniform float wavePhase[8];
uniform int n_waves;
uniform float time;

void main() {
	vec4 Vertex0 = mix(trap[0], trap[1], texPos.x);
	vec4 Vertex1 = mix(trap[3], trap[2], texPos.x);
	vec4 Vertex = mix(Vertex0, Vertex1, texPos.y);
	Vertex /= Vertex.w;
	float term = 0;
	for (int i = 0; i < n_waves; ++i) {
		term += ampl[i] * cos(dot(waveVector[i],Vertex.xyz) - time * sqrt(9.8* 2 * 3.1415 / waveLength[i])  + wavePhase[i]);
	}
	Vertex.z = term;
	gl_Position = m_mvp * Vertex;
}