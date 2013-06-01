#version 400
#define MAX_WAVE_RESOLUTION 32

layout (location = 0) out vec4 col;

uniform sampler2D sky;
uniform float lx;
uniform float lz;
uniform vec3 camera;
uniform vec3 sun_direction;
uniform vec3 c0;
uniform vec3 c90;
uniform vec3 specular;
uniform float specular_strength;
uniform float specular_power;
uniform bool geometry;	


in vec3 worldPos;
in vec3 tex_data;

void main() {	
	float sky_height = 10;
	vec3 norm = normalize(vec3(-tex_data.y, -tex_data.z, 1.0));
	//vec3 norm = vec3(0, 0, 1.f);
	vec3 a = normalize(camera - worldPos);
	vec3 ni = normalize(sun_direction);
	vec3 reflec = 2 * norm * dot(norm, ni) - ni; 
	vec3 term = -reflec - worldPos;
	//col = vec4(color_sky, 1.0);
	vec3 color_sky = texture(sky, vec2(term.x * sky_height / term.z, term.y * sky_height / term.z)).xyz;
	float p = asin(length(ni * norm));
	float q = asin(sin(p) * 0.75);
	float r_frenel = 0.5 * (sin(q - p) * sin(q - p) / (sin(p + q) * sin(p + q)) + tan(q - p) * tan(q - p) / (tan(q + p) * tan(q + p)));
	//float r_frenel = 1 / pow(1 + cos(p), 8);
	vec3 c = mix(c90, c0, abs(dot(a, norm))) * (1 - r_frenel) + color_sky * r_frenel;
	vec3 h = normalize(ni + a);
	float dist = length(camera - worldPos);
	c = mix(c, specular_strength * pow(abs(dot(norm, h)), specular_power) * specular * pow(dist, .1), 0.05);
	if (geometry) {
		col = vec4(c, 1.);
	} else {
		col = vec4(0,0,1, 0.8);
	}
}