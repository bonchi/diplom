#version 400

#define MAX_WAVE_RESOLUTION 32

layout(triangles, equal_spacing) in;

uniform mat4 m_mvp;
uniform float lx;
uniform float lz;
uniform sampler2D tex_tex;

out vec3 worldPos;
out vec3 tex_data;

void main(void)
{
	vec4 Vertex = gl_TessCoord.x * gl_in [0].gl_Position + 
				gl_TessCoord.y * gl_in [1].gl_Position +
				gl_TessCoord.z * gl_in [2].gl_Position;
	Vertex /= Vertex.w;
	vec2 term;
	float kx = - (MAX_WAVE_RESOLUTION - 1) * lx / MAX_WAVE_RESOLUTION;
	float kz = - (MAX_WAVE_RESOLUTION - 1) * lz / MAX_WAVE_RESOLUTION;
	term.x = Vertex.x - int(Vertex.x / kx) * kx;
	if (term.x > 0) {
		term.x += kx;
	}
	term.y = Vertex.y - int(Vertex.y / kz) * kz;
	if (term.y > 0) {
		term.y += kz;
	}
	tex_data = texture(tex_tex, vec2(term.x / kx, term.y / kz)).xyz;
	Vertex.z = tex_data.x;
	worldPos = Vertex.xyz;
	gl_Position = m_mvp * Vertex;
}