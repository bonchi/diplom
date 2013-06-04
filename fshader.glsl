#version 400

layout (location = 0) out vec4 col;

uniform sampler2D sky;
uniform float lx;
uniform float lz;
uniform vec3 camera;
uniform vec3 sun_direction;
uniform vec4 c0;
uniform vec4 c90;
uniform vec3 specular;
uniform float specular_strength;
uniform float specular_power;
uniform bool geometry;	


in vec3 worldPos;
in vec3 tex_data;
in vec2 koord;

void main() {	
	float distanceAttenuation = 0.05; 
	float sky_height = 10;
	vec3 norm = normalize(vec3(-tex_data.y, -tex_data.z, 1.0));
	float distS = exp(-length(camera - worldPos) * distanceAttenuation);
	norm = normalize(mix(vec3(0, 0, 1.f), norm, distS));
	//vec3 norm = vec3(0, 0, 1.f);
	vec3 a = normalize(camera - worldPos);
	vec3 sun_dir = normalize(sun_direction);
	vec3 reflec = 2 * norm * dot(norm, a) - a; 
	vec4 color_sky = vec4(texture(sky, 0.01 * vec2(reflec.x / reflec.z,  reflec.y / reflec.z)).xyz, 1);
	float p = acos(dot(a, norm));
	float q = asin(sin(p) * 0.75);
	float r_frenel = 0.5 * (sin(p - q) * sin(p - q) / (sin(p + q) * sin(p + q)) + tan(p - q) * tan(p - q) / (tan(q + p) * tan(q + p)));
	vec4 c = mix(c90, c0, abs(dot(a, norm))) * (1 - r_frenel) + color_sky * r_frenel;	
	vec3 ia = vec3(0.05, 0.05, 0.05);
	vec3 id = dot(sun_dir, norm) * vec3(0.05, 0.10, 0.10);
	vec3 h = normalize(sun_dir + a);
	vec3 is = specular_strength * pow(abs(dot(norm, h)), specular_power) * specular;
	vec3 i = ia + id + is;
	if (i.x < 0) i.x = 0;
	if (i.y < 0) i.y = 0;
	if (i.z < 0) i.z = 0;
	if (i.x > 1) i.x = 1;
	if (i.y > 1) i.y = 1;
	if (i.z > 1) i.z = 1;
	c = c + vec4(i, 1.0);
	if (geometry) {
		col = c;
	} else {
		col = vec4(0,0,1, 1.);
	}
	//col = vec4(fract(koord), 0.0, 1.0);
}