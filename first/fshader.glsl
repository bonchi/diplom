#version 330

layout (location = 0) out vec4 col;

in vec3 worldPos;

uniform vec3 sun_c;
uniform vec3 sun_direction;
uniform vec3 camera_pos;

void main()
{	
	vec3 diffuse = vec3(0.5, 0.05, 0.3);
    vec3 specular = vec3(0.1, 0.1, 0.5);
	//vec3 ambient = vec3(0, 0, 0.1);
    vec3 cameraDir = normalize(camera_pos - worldPos);
    vec3 normal = normalize(vec3(0, 0.3, 1));
	
	vec3 reflection = normalize(reflect(-sun_direction, normal));
    vec3 Kspecular = specular * sun_c * max(dot(cameraDir, reflection), 0.0);
    vec3 Kdiffuse =  diffuse * sun_c * max(dot(sun_direction, normal),0.0);

    vec3 light = Kspecular + Kdiffuse;
    col = vec4(light, 1.0);
}