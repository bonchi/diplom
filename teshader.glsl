#version 420

layout(triangles, equal_spacing, ccw ) in;

uniform mat4 m_mvp;
uniform float lx;
uniform float lz;
uniform sampler2D h_field;
uniform sampler2D normal_x_field;
uniform sampler2D normal_y_field;
uniform int wave_res;

in tc_output {
	vec2 pos;
	vec3 koord_den;
	float density;
} te_in[];

out te_output {
	vec3 koord_den;
	float density;
} te_out;

out vec3 worldPos;
out vec3 tex_data;
out vec2 koord;
out vec2 Koord_d;

void main(void)
{
	vec4 Vertex = gl_TessCoord.x * gl_in [0].gl_Position + 
				gl_TessCoord.y * gl_in [1].gl_Position +
				gl_TessCoord.z * gl_in [2].gl_Position;
	Vertex /= Vertex.w;
	float kx = - (wave_res - 1) * lx / wave_res;
	float kz = - (wave_res - 1) * lz / wave_res;
	koord = vec2(Vertex.x / kx, Vertex.y / kz);
	tex_data = vec3(texture(h_field, koord).x, texture(normal_x_field, koord).x, texture(normal_y_field, koord).x);
	Vertex.z = tex_data.x;
	worldPos = Vertex.xyz;
	te_out.koord_den = te_in[0].koord_den;
	te_out.density = te_in[0].density;
	gl_Position = m_mvp * Vertex;
}