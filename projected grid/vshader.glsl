#version 330

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
	if (Vertex.x  > 0) {
		term.x = Vertex.x - lx * (int(Vertex.x / lx));
	} else {
		term.x = lx * (abs(int(Vertex.x / lx)) + 1) + Vertex.x;
	}
	if (Vertex.y  > 0) {
		term.y = Vertex.y - lz * (int(Vertex.y / lz));
	} else {
		term.y = lz * (abs(int(Vertex.y / lz)) + 1) + Vertex.y;
	}
	Vertex.z = texture(tex_tex, vec2(term.x / lx, term.y / lz)).x;
	gl_Position = m_mvp * Vertex;
}