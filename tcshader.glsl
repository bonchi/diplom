#version 420

#define ID gl_InvocationID

layout ( vertices = 3 ) out;  

uniform sampler2D density;   
uniform mat4 m_camera;
uniform float lx;
uniform float lz;
uniform sampler2D h_field;
uniform int wave_res;
uniform int number_level;

patch uint level;

out float diff[];
out float tin[];
out float tout[];
out vec4 Vertex[];
out float dist_sc[];

out tc_output {
	vec2 pos;
	vec3 koord_den;
	float density;
} tc_out[];

float get_value(int x, int y, uint level) {
	int newx = x % wave_res;
	int newy = y % wave_res;
	if (newx < 0)
		newx += wave_res;
	if (newy < 0)
		newy += wave_res;

	newx >>= int(level);
	newy >>= int(level);
	tc_out[ID].koord_den = vec3(vec2(newx / float(wave_res >> level), newy / float(wave_res >> level)), float(level));
	tc_out[ID].density = textureLod(density, vec2(newx / float(wave_res >> level), newy / float(wave_res >> level)), level).x;
	return tc_out[ID].density;
}

void main ()
{           
	gl_TessLevelOuter[ID] = 1;
	gl_out[ID].gl_Position = gl_in[ID].gl_Position;
	float kx = - (wave_res - 1) * lx / (wave_res);
	float kz = - (wave_res - 1) * lz / (wave_res);
	vec2 v = gl_in[ID].gl_Position.xy;
	tc_out[ID].pos.x =  (wave_res - 1) * v.x / kx;
	tc_out[ID].pos.y =  (wave_res - 1) * v.y / kz;
	tc_out[ID].koord_den = vec3(1);
	tc_out[ID].density = 0;
	barrier();
	int i1 = ID;
	int i2 = ID + 1;
	if (i1 == 2)
		i2 = 0;

	int q1x = int(floor(tc_out[i1].pos.x));
	int q1y = int(floor(tc_out[i1].pos.y));
	int q2x = int(floor(tc_out[i2].pos.x));
	int q2y = int(floor(tc_out[i2].pos.y));

	tin[ID] = 1;
	diff[ID] = 0;
	Vertex[ID] = gl_in [ID].gl_Position;
	Vertex[ID] /= Vertex[ID].w;
	Vertex[ID].z = texture(h_field, vec2(Vertex[ID].x / kx, Vertex[ID].y / kz)).x;
	Vertex[ID] = m_camera * Vertex[ID];
	barrier();
	dist_sc[ID] = distance(Vertex[i1], Vertex[i2]);
	
	diff[ID] = max(diff[ID], distance(tc_out[i1].pos, tc_out[i2].pos));

	barrier();

	//tout[ID] = 5 * get_value(q1x, q1y, uint(min(int(ceil(log(diff[ID]))), number_level - 1)));
	tout[ID] = log(diff[ID]);

	if (ID == 0) {
		float term = max(diff[0], diff[1]);
		term = max(term, diff[2]);
		level = uint(min(int(ceil(log(term))), number_level - 1));
		term = max(dist_sc[0], dist_sc[1]);
		term = max(term, dist_sc[2]);
		tin[ID] = 5 * term * get_value(q1x, q1y, level);
		//tin[ID] = 5 * get_value(q1x, q1y, level);
	}
	barrier();
	gl_TessLevelInner[0] = min(tin[0], 5);
	gl_TessLevelOuter[ID] = max(tout[i2], 1);
	gl_out[ID].gl_Position = gl_in [ID].gl_Position;
}