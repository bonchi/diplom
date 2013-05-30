#version 330
#define MAX_WAVE_RESOLUTION 64

layout (location = 0) out vec4 col;

uniform sampler2D tex_tex;
uniform float lx;
uniform float lz;

in vec4 worldPos;

void main()
{	
	vec2 term;
	float kx = - (MAX_WAVE_RESOLUTION - 1) * lx / MAX_WAVE_RESOLUTION;
	float kz = - (MAX_WAVE_RESOLUTION - 1) * lz / MAX_WAVE_RESOLUTION;
	term.x = worldPos.x - int(worldPos.x / kx) * kx;
	if (term.x > 0) {
		term.x += kx;
	}
	term.x = term.x / kx;
	term.y = worldPos.y - int(worldPos.y / kz) * kz;
	if (term.y > 0) {
		term.y += kz;
	}
	term.y = term.y / kz;
	//col = texture(tex_tex, vec2(term.x, term.y));
	col = vec4(fract(term), 0, 1);
	//col = vec4(0, 0, 1, 1);
}