#version 400

#define ID gl_InvocationID

layout ( vertices = 3 ) out;  

uniform sampler2D density;     
uniform float lx;
uniform float lz;
uniform sampler2D tex_tex;
uniform int wave_res;
uniform float inner_big_part;
uniform float outer_big_part;
uniform float koef_inner_density;
uniform float koef_outter_density;
patch bool flag;

out vec2 pos[3];
out vec4 min_max[3];
out float tout[3];
out float tin[3];

bool viewed(int i, int j, int i1) {
	if (i1 == 0) return false;
	if ((i >= min_max[0].x) && (i <= min_max[0].y) && (j >= min_max[0].z) && (j <= min_max[0].w)) {
		if ((min_max[0].y - min_max[0].x <= wave_res - 1) && (min_max[0].w - min_max[0].z) <= wave_res - 1) {
			return true;
		}
		return false;
	}
	if (i1 == 1) return false;
	if ((i >= min_max[1].x) && (i <= min_max[1].y) && (j >= min_max[1].z) && (j <= min_max[1].w)) {
		if ((min_max[1].y - min_max[1].x <= wave_res - 1) && (min_max[1].w - min_max[1].z) <= wave_res - 1) {
			return true;
		}
		return false;
	}
	return false;
}

float vec_mult(vec2 a, vec2 b) {
	return a.x * b.y - a.y * b.x;
}

bool check_point(int i, int j, int i1, int i2, int i3) {
	//vec3 equations[3];

	vec2 a = -pos[i1] + pos[i2];
	vec2 b = -pos[i1] + pos[i3];
	vec2 c = -pos[i2] + pos[i3];
	vec2 x1 = -pos[i1] + vec2(i, j);
	vec2 x2 = -pos[i2] + vec2(i, j);
	vec2 x3 =-pos[i3] + vec2(i, j);
	if ((vec_mult(a, b) * vec_mult(a, x1) < 0) || (vec_mult(c, -a) * vec_mult(c, x2) < 0) || (vec_mult(-b, -c) * vec_mult(-b, x3) < 0)) {
		return false;
	}
	return true;
}

int check_q(int i, int j, int i1, int i2) {
	int i3 = i2 + 1;
	if (i3 == 3) i3 = 0;
	bool l1 = check_point(i, j, i1, i2, i3);
	bool l2 = check_point(i, j + 1, i1, i2, i3);
	bool l3 = check_point(i + 1, j, i1, i2, i3);
	bool l4 = check_point(i + 1, j + 1, i1, i2, i3);
	if (l1 && l2 && l3 && l4) return 2;
	if (l1 || l2 || l3 || l4) return 1;
	return 0;
}

void main ()
{           
	/*gl_out [ID].gl_Position = gl_in [ID].gl_Position;
	gl_TessLevelOuter[ID] = 1;
	gl_TessLevelInner[0] = 1;
	barrier();
    return;*/
	flag = false;
	float kx = - (wave_res - 1) * lx / (wave_res);
	float kz = - (wave_res - 1) * lz / (wave_res);
	vec2 Vertex = gl_in[ID].gl_Position.xy;
	pos[ID].x =  wave_res * Vertex.x / kx;
	pos[ID].y =  wave_res * Vertex.y / kz;
	barrier();
	vec3 equation;
	int i1 = ID;
	int i2 = ID + 1;
	if (i1 == 2) {
		i2 = 0;
	}
	int q1x = int(pos[i1].x);
	if (pos[i1].x < 0) {
		--q1x;
	}
	int q1y = int(pos[i1].y);
	if (pos[i1].y < 0) {
		--q1y;
	}
	int q2x = int(pos[i2].x);
	if (pos[i2].x < 0) {
		--q2x;
	};
	int q2y = int(pos[i2].y);
	if (pos[i2].y < 0) {
		--q2y;
	}
	min_max[ID] = vec4(min(q1x, q2x), max(q1x, q2x), min(q1y, q2y), max(q1y, q2y));
	tin[ID] = 0;
	barrier(); 
	tout[ID] = 1;
	if ((min_max[i1].y - min_max[i1].x > wave_res - 1) || (min_max[i1].w - min_max[i1].z) > wave_res - 1) {
		tout[ID] = outer_big_part * max((min_max[i1].w - min_max[i1].z) / (wave_res - 1), (min_max[i1].y - min_max[i1].x) / (wave_res - 1));
		tin[ID] = inner_big_part * tout[ID];
		flag = true;
	} else { 
		for (int i = int(min_max[i1].x); i <= int(min_max[i1].y); ++i) {
			for (int j = int(min_max[i1].z); j <= int(min_max[i1].w); ++j) {
				int t = check_q(i, j, i1, i2);
				if (t == 1) {
					tout[ID] += koef_outter_density * texture(density, vec2(i / (wave_res - 1), j / (wave_res - 1))).x;
				} else if (t == 2) {
					if (viewed(i, j, i1)) continue;
					tin[ID] += koef_inner_density * texture(density, vec2(i / (wave_res - 1), j / (wave_res - 1))).x;
				}
			}
		}
	}
	barrier();
	gl_TessLevelOuter[ID] = tout[i2];
	barrier();
	gl_TessLevelInner[0] = tin[0] + tin[1] + tin[2] + 1;
	gl_out [ID].gl_Position = gl_in [ID].gl_Position;
}