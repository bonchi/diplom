#version 400

layout(triangles, equal_spacing, ccw ) in;

uniform mat4 m_mvp;
uniform float lx;
uniform float lz;
uniform sampler2D tex_tex;
uniform int wave_res;

out vec3 worldPos;
out vec3 tex_data;
out vec2 koord;

void main(void)
{
	vec4 Vertex = gl_TessCoord.x * gl_in [0].gl_Position + 
				gl_TessCoord.y * gl_in [1].gl_Position +
				gl_TessCoord.z * gl_in [2].gl_Position;
	Vertex /= Vertex.w;
	float kx = - (wave_res - 1) * lx / wave_res;
	float kz = - (wave_res - 1) * lz / wave_res;
	koord = vec2(Vertex.x / kx, Vertex.y / kz);
	tex_data = texture(tex_tex, vec2(Vertex.x / kx, Vertex.y / kz)).xyz;
	Vertex.z = tex_data.x;
	worldPos = Vertex.xyz;
	gl_Position = m_mvp * Vertex;
}