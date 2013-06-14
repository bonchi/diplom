#include <memory>
#include "vars_and_constants.h"

struct cube_map_t {
	cube_map_t(std::string const& dir) {
		glGenTextures(1, &tex_);

		glTextureParameteriEXT(tex_, GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTextureParameteriEXT(tex_, GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

		for (int i = 0; i < NUM_LAYERS; i++)
			load_cube_map(dir + "\\" + layer_suf[i] + ".jpg", layer[i]);
	}

	void bind(GLenum tex_unit) {
		glActiveTexture(tex_unit);
		glBindTexture(GL_TEXTURE_CUBE_MAP, tex_);
	}

	~cube_map_t() {
		glDeleteTextures(1, &tex_);
	}

private:
	void load_cube_map(std::string const& file_name, GLenum layer) {
		Image cm_layer;
		cm_layer.read(file_name);
		Blob blob;
		cm_layer.write(&blob, "RGB", 8);
		glTextureImage2DEXT(tex_, layer, 0, GL_RGB32F, cm_layer.columns(), cm_layer.rows(), 0, GL_RGB, GL_UNSIGNED_BYTE, blob.data());
	}

	static const int NUM_LAYERS = 6;
	static char const * layer_suf[NUM_LAYERS];
	static GLenum layer[NUM_LAYERS];

private:
	GLuint tex_;
};

char const * cube_map_t::layer_suf[cube_map_t::NUM_LAYERS] = { "negx", "negy", "negz", "posx", "posy", "posz" };
GLenum cube_map_t::layer[cube_map_t::NUM_LAYERS] = {  
	GL_TEXTURE_CUBE_MAP_NEGATIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 
	GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_POSITIVE_Z };

namespace {
	std::shared_ptr<cube_map_t> env_cm;
}

vec3 buildProjectorPos2(const vec3 &camera_pos) {
	float x = camera_pos.x;
	float y = camera_pos.y;
	float z = camera_pos.z;
	if (z < 0) {
		z = -z;
	}
	if (z < SUPP) {
		z = SUPP;
	}
	return vec3(x, y, z);
}

void intersection(vec4 a, vec4 b, float h, vec4 * trap, int & count) {
	vec4 term = b - a;
	if (term.z - term.w * h != 0) {
		float k = (a.w * h - a.z) / (term.z - term.w * h);
		vec4 p = a + k * term;
		p /= p.w;
		if ((b.z / b.w - p.z) * (a.z / a.w - p.z) <= 0) {
			trap[count] = p;
			trap[count].z = 0;
			trap[count] /= trap[count].w;		
			++count;
		}
	}
}

void idle() {
	glutPostRedisplay();
}

void display() {
	glUseProgram(0);
	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	mat4 m = c_main.mvp();
	mat4 im = inverse(m);
	mat4 m2 = c_sec.mvp();
	vec4 rc[8];
	vec4 rc2[8];
	vec4 rc3[8];

	for (int i = 0; i < 4; ++i) {
		rc[2 * i] = im * cube[2 * i];
		rc[2 * i] = m2 * rc[2 * i];
	}
	glColor3f(1.0f, 0, 0);
	glLineWidth(2.0);
	vec4 camera_vert = m2 * vec4(c_main.pos(), 1.0);
	for (int i = 0; i < 4; ++i) {
		glBegin(GL_LINES);
			glVertex4f(camera_vert.x, camera_vert.y, camera_vert.z, camera_vert.w);
			glVertex4f(rc[2 * i].x, rc[2 * i].y, rc[2 * i].z, rc[2 * i].w);
		glEnd();			
	}
	glLineWidth(1.0);

	vec3 point  = c_main.pos() + normalize(c_main.dir()) * DIST;
	point.z = 0;
	vec3 cam_dir = c_main.dir();
	vec3 pj_pos = buildProjectorPos2(c_main.pos());
	
	mat4 m_pview = lookAt(pj_pos, point, vec3(0, 0, 1));
	mat4 m_proj = inverse(c_main.perm() * m_pview);

	for (int i = 0; i < 8; ++i) {
		rc[i] = im * cube[i];
		rc2[i] = im * cube2[i];
		rc3[i] = im * cube3[i];
	}
	vec4 trap[32];
	int count = 0;
	for (int i = 0; i < 4; ++i) {
		//боковые грани
		intersection(rc[2 * i + 1], rc[2 * i], SUPP, trap, count);
		intersection(rc[2 * i + 1], rc[2 * i], SLOW, trap, count);
		//точки
		if ((rc[2 * i].z / rc[2 * i].w < SUPP) && (rc[2 * i].z / rc[2 * i].w > SLOW)) {
			trap[count] = rc[2 * i];
			trap[count].z = 0;
			trap[count] /= trap[count].w;
			++count;
		}
		if ((rc[2 * i + 1].z / rc[2 * i + 1].w < SUPP) && (rc[2 * i + 1].z / rc[2 * i + 1].w > SLOW)) {
			trap[count] = rc[2 * i + 1];
			trap[count].z = 0;
			trap[count] /= trap[count].w;
			++count;
		}
		//пересечения для ближней и дальней плоскостей
		intersection(rc2[2 * i + 1], rc2[2 * i], SUPP, trap, count);
		intersection(rc2[2 * i + 1], rc2[2 * i], SLOW, trap, count);
		intersection(rc3[2 * i + 1], rc3[2 * i], SUPP, trap, count);
		intersection(rc3[2 * i + 1], rc3[2 * i], SLOW, trap, count);
	}	
	
	if (count > 0) {
		mat4 tr = inverse(m_proj);
		for (int i = 0; i < count; ++i) {
			trap[i] = tr * trap[i];
			trap[i] /= trap[i].w;
		}
		float xmin, xmax, ymin, ymax;
		xmin = trap[0].x;
		xmax = trap[0].x;
		ymin = trap[0].y;
		ymax = trap[0].y;
		for (int i = 1; i < count; ++i) {
			if (trap[i].x < xmin) {
				xmin = trap[i].x;
			}
			if (trap[i].x > xmax) {
				xmax = trap[i].x;
			}
			if (trap[i].y < ymin) {
				ymin = trap[i].y;
			}
			if (trap[i].y > ymax) {
				ymax = trap[i].y;
			}
		}

		mat4 m_range = mat4(vec4(xmax - xmin, 0, 0, 0),vec4(0, ymax - ymin, 0, 0), vec4(0, 0, 1, 0), vec4(xmin, ymin, 0, 1));
		mat4 m_proj2 =  m_proj * m_range;
		glUseProgram(prg);
		if (!pause) {
			generation_h(0.5 * (float) clock() / CLOCKS_PER_SEC);
		}

		cudaArray * hf = NULL;
		cudaArray * hf_nx = NULL;
		cudaArray * hf_ny = NULL;

		cudaError_t l;

		l = cudaGraphicsMapResources(1, &resource1, 0);
		l = cudaGraphicsMapResources(1, &resource2, 0);
		l = cudaGraphicsMapResources(1, &resource3, 0);
		l = cudaGraphicsMapResources(1, &resource4, 0);

		l = cudaGraphicsSubResourceGetMappedArray (&hf,  resource1, 0, 0);
		l = cudaGraphicsSubResourceGetMappedArray (&hf_nx, resource2, 0, 0);
		l = cudaGraphicsSubResourceGetMappedArray (&hf_ny, resource3, 0, 0);

		do_fft(hf, hf_nx, hf_ny, resource4);

		l = cudaGraphicsUnmapResources(1, &resource1, 0);
		l = cudaGraphicsUnmapResources(1, &resource2, 0);
		l = cudaGraphicsUnmapResources(1, &resource3, 0);
		l = cudaGraphicsUnmapResources(1, &resource4, 0);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, h_field);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, normal_x_field);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, normal_y_field);
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, tex_sky);
		glActiveTexture(GL_TEXTURE4);
		glBindTexture(GL_TEXTURE_2D, density);
		glBindBuffer(GL_ARRAY_BUFFER, buf_tex);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buf_index);
		//env_cm->bind(GL_TEXTURE5);
		

		glPatchParameteri ( GL_PATCH_VERTICES, 3);
		glDrawElements(GL_PATCHES, index.size(), GL_UNSIGNED_INT, NULL);

		for (int i = 0; i < 8; ++i) {
			rc[i] = m_proj2 * cube[i];
		}
		for (int i = 0; i < 4; ++i) {
			vec4 term = rc[2 * i] - rc[2 * i + 1];
			float k = -rc[2 * i + 1].z / term.z;
			trap[i] = rc[2 * i + 1] + k * term;
			float term2 = 0;
		}

		vec3 pos_c = c_main.pos();

		glUniform1f(glGetUniformLocation(prg, "inner_big_part"), inner_big_part);
		glUniform1f(glGetUniformLocation(prg, "outer_big_part"), outer_big_part);
		glUniform1f(glGetUniformLocation(prg, "koef_inner_density"), koef_inner_density);
		glUniform1f(glGetUniformLocation(prg, "koef_outter_density"), koef_outter_density);
		glUniform1f(glGetUniformLocation(prg, "inner_level"), inner_level);
		glUniform1f(glGetUniformLocation(prg, "outer_level"), outer_level);
		glUniform1i(glGetUniformLocation(prg, "h_field"), 0);
		glUniform1i(glGetUniformLocation(prg, "normal_x_field"), 1);
		glUniform1i(glGetUniformLocation(prg, "normal_y_field"), 2);
		glUniform1i(glGetUniformLocation(prg, "sky"), 3);
		glUniform1i(glGetUniformLocation(prg, "density"), 4);
		glUniform3fv(glGetUniformLocation(prg, "sun_direction"), 1, value_ptr(sun_direction));	
		glUniform3fv(glGetUniformLocation(prg, "camera"), 1, value_ptr(pos_c));
		glUniformMatrix4fv(glGetUniformLocation(prg, "m_mvp"), 1, false, value_ptr(m2));
		glUniformMatrix4fv(glGetUniformLocation(prg, "m_camera"), 1, false, value_ptr(m));
		glUniform4fv(glGetUniformLocation(prg, "trap"), 4, value_ptr(trap[0]));
		glUniform1f(glGetUniformLocation(prg, "lx"), lx);
		glUniform1f(glGetUniformLocation(prg, "lz"), lz);
		glUniform4fv(glGetUniformLocation(prg, "c0"), 1, value_ptr(c0));	
		glUniform4fv(glGetUniformLocation(prg, "c90"), 1, value_ptr(c90));
		//glUniform3fv(glGetUniformLocation(prg, "sky"), 1, value_ptr(sky));
		glUniform3fv(glGetUniformLocation(prg, "specular"), 1, value_ptr(specular));
		glUniform1f(glGetUniformLocation(prg, "specular_strength"), specular_strength);
		glUniform1f(glGetUniformLocation(prg, "specular_power"), specular_power);
		glUniform1i(glGetUniformLocation(prg, "geometry"), geometry);
		glUniform1i(glGetUniformLocation(prg, "wave_res"), MAX_WAVE_RESOLUTION);
		glUniform1i(glGetUniformLocation(prg, "number_texel"), number_texel);
		glUniform1i(glGetUniformLocation(prg, "number_level"), (int)log2(1.f * MAX_WAVE_RESOLUTION) + 1);
		glUniform1i(glGetUniformLocation(prg, "tex_env"), 5);
	}
	TwDraw();
	glFlush();
	glutSwapBuffers();
}

void generationgrid(int x0, int x1, int y0, int y1, int wid, int cache_size, std::vector <int> &ind) {
	if (x1 - x0 + 1 <= cache_size - 1) {
		for (int x = x0; x <= x1; ++x) {
			ind.push_back(y0 * wid + x);
			ind.push_back(y0 * wid + x);
			ind.push_back(y0 * wid + x);
		}
		for (int y = y0; y < y1; ++y) {
			for (int x = x0; x < x1; ++x) {
				ind.push_back(y * wid + x);
				ind.push_back((y + 1) * wid + x);
				ind.push_back(y * wid + x + 1);

				ind.push_back((y + 1) * wid + x);
				ind.push_back(y * wid + x + 1);
				ind.push_back((y + 1) * wid + x + 1);
			}
		}
	} else {
		int x_new = x0 + cache_size - 2;
		generationgrid(x0, x_new, y0, y1, wid, cache_size, ind);
		generationgrid(x_new, x1, y0, y1, wid, cache_size, ind);
	}
}

void rebuildGrid() {
	 pos = new float [2 * (max_resolution + 1) * (max_resolution + 1)];
	for (int i = 0; i <= resolution; ++i) {
		for (int j = 0; j <= resolution; ++j) {
			pos[2 * (i * (resolution + 1) + j)] = ((float) i) / resolution;
			pos[2 * (i * (resolution + 1) + j) + 1] = ((float) j) / resolution;
		}
	}
	index.resize(0);
	generationgrid(0, resolution, 0, resolution, resolution + 1, 8, index);
	glNamedBufferDataEXT(buf_tex, 2 * (resolution + 1) * (resolution + 1) * sizeof(float), pos, GL_STATIC_DRAW);
	int t = index.size();
	int* ind = new int[t];
	for (int i = 0; i < t; ++i) {
		ind[i] = index[i];
	}
	glNamedBufferDataEXT(buf_index, t * sizeof(int), ind, GL_STATIC_DRAW);
}



void init() {

	glGenTextures(1, &h_field);
	glBindTexture(GL_TEXTURE_2D, h_field);
	glTextureParameteriEXT(h_field, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTextureParameteriEXT(h_field, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, MAX_WAVE_RESOLUTION, MAX_WAVE_RESOLUTION, 0, GL_RG, GL_FLOAT, NULL);

	glGenTextures(1, &normal_x_field);
	glBindTexture(GL_TEXTURE_2D, normal_x_field);
	glTextureParameteriEXT(normal_x_field, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTextureParameteriEXT(normal_x_field, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, MAX_WAVE_RESOLUTION, MAX_WAVE_RESOLUTION, 0, GL_RG, GL_FLOAT, NULL);

	glGenTextures(1, &normal_y_field);
	glBindTexture(GL_TEXTURE_2D, normal_y_field);
	glTextureParameteriEXT(normal_y_field, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTextureParameteriEXT(normal_y_field, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, MAX_WAVE_RESOLUTION, MAX_WAVE_RESOLUTION, 0, GL_RG, GL_FLOAT, NULL);

	glGenTextures(1, &density);
	glBindTexture(GL_TEXTURE_2D, density);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, MAX_WAVE_RESOLUTION, MAX_WAVE_RESOLUTION, 0, GL_RED, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);  
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glGenerateMipmap(GL_TEXTURE_2D);
 
	cudaError_t l;
	l = cudaGraphicsGLRegisterImage(&resource1, h_field, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
	l = cudaGraphicsGLRegisterImage(&resource2, normal_x_field, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
	l = cudaGraphicsGLRegisterImage(&resource3, normal_y_field, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
	l = cudaGraphicsGLRegisterImage(&resource4, density, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);

	generation_h0(lx, lz, wind.x, wind.y, A_norm);
	glGenBuffers(1, &buf_tex);
	glGenBuffers(1, &buf_index);
	rebuildGrid();
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	glBindBuffer(GL_ARRAY_BUFFER, buf_tex);
	glVertexAttribPointer(1, 2, GL_FLOAT, false, 0, NULL);
	glEnableVertexAttribArray(1);	
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	Image sky;
	Blob blob;
	sky.read("sky1.bmp" );
	sky.write(&blob, "RGB", 8);

	//env_cm.reset(new cube_map_t("C:\\Users\\kaa\\Downloads\\Pond"));

	glGenTextures(1, &tex_sky);
	glBindTexture(GL_TEXTURE_2D, tex_sky);
	glTextureParameteriEXT(tex_sky, GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glTextureParameteriEXT(tex_sky, GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, sky.columns(), sky.rows(), 0, GL_RGB, GL_UNSIGNED_BYTE, blob.data());
	glEnable(GL_TEXTURE_2D);
	if (geometry) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	} else {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	}

	GLuint fshader = glCreateShader(GL_FRAGMENT_SHADER);
	GLuint vshader = glCreateShader(GL_VERTEX_SHADER);
	GLuint tcshader = glCreateShader(GL_TESS_CONTROL_SHADER);
	GLuint teshader = glCreateShader(GL_TESS_EVALUATION_SHADER);
	int size;
	{
		std::ifstream stin("vshader.glsl");
		std::string source;
		while (stin)
		{
			std::string line;
			getline(stin, line);
			source += line;
			source += "\n";
		}
		char const * buffer = source.c_str();
		size = source.length();
		glShaderSource(vshader, 1, (const GLchar **)&buffer, &size);
	}
	{
		std::ifstream stin("fshader.glsl");
		std::string source;
		while (stin)
		{
			std::string line;
			getline(stin, line);
			source += line;
			source += "\n";
		}
		char const * buffer = source.c_str();
		size = source.length();
		glShaderSource(fshader, 1, (const GLchar **)&buffer, &size);
	}
	{
		std::ifstream stin("tcshader.glsl");
		std::string source;
		while (stin)
		{
			std::string line;
			getline(stin, line);
			source += line;
			source += "\n";
		}
		char const * buffer = source.c_str();
		size = source.length();
		glShaderSource(tcshader, 1, (const GLchar **)&buffer, &size);
	}
	{
		std::ifstream stin("teshader.glsl");
		std::string source;
		while (stin)
		{
			std::string line;
			getline(stin, line);
			source += line;
			source += "\n";
		}
		char const * buffer = source.c_str();
		size = source.length();
		glShaderSource(teshader, 1, (const GLchar **)&buffer, &size);
	}
	glCompileShader(vshader);
	GLint param;

	glCompileShader(fshader);
	glCompileShader(tcshader);
	glCompileShader(teshader);

	prg = glCreateProgram();
	glAttachShader(prg, vshader);
	glAttachShader(prg, fshader);
	glAttachShader(prg, tcshader);
	glAttachShader(prg, teshader);
	glLinkProgram(prg);

	glGetProgramiv(prg, GL_LINK_STATUS, &param);
	{
		GLint len;
		glGetProgramiv(prg, GL_INFO_LOG_LENGTH, &len);
		char * buff = new char[len];
		glGetProgramInfoLog(prg, len, &len, buff);
		std::cerr << buff << std::endl;
		delete [] buff;
	}
	
	//if (!param)
	//	throw 1;
}

void reshape(int width, int height) {
   TwWindowSize(width, height);
   glViewport(0, 0, width, height);
   c_sec.set_aspect(1.f * width / height);
}

bool flag = true;

void motionMouse(int x, int y) {
	if (TwEventMouseMotionGLUT(x, y)) return;

	if (flag) {
		c_sec.motionMouse(x, y);
	} else {
		c_main.motionMouse(x, y);
	}
}

void mouse(int button, int state, int x, int y) {
	if (TwEventMouseButtonGLUT(button, state, x, y)) return;

	if (flag) {
		c_sec.mouse(button, state, x, y);
	} else {
		c_main.mouse(button, state, x, y);
	}
}

void key(unsigned char k, int x, int y) {
	if (TwEventKeyboardGLUT(k, x, y)) {
		rebuildGrid();
		return;
	}

	if (k == 'Q') {
		exit(0);
	} 
	if (k == 9) {
		flag = !flag;
	}
	if (flag) {
		c_sec.key(k);
	} else {
		c_main.key(k);
	}
	display();

}

struct Point { 
	float X, Y, Z; 
	Point(vec3 a) {
		X = a.x;
		Y = a.y;
		Z = a.z;
	}
};

void TW_CALL camera_get_pos (void *value, void *clientData) {
	*static_cast<Point *>(value) = (static_cast<Camera *>(clientData)) ->pos();
}

void TW_CALL camera_get_fov (void *value, void *clientData) {
	*static_cast<float *>(value) = (static_cast<Camera *>(clientData)) -> getFovy();
}

void TW_CALL camera_get_rotation (void *value, void *clientData) {
	*static_cast<quat *>(value) = (static_cast<Camera *>(clientData)) -> getCameraRotation();
}

void TW_CALL camera_get_roll (void *value, void *clientData) {
	*static_cast<float *>(value) = (static_cast<Camera *>(clientData)) -> getRoll();
}

void TW_CALL camera_get_pitch (void *value, void *clientData) {
	*static_cast<float *>(value) = (static_cast<Camera *>(clientData)) -> getPitch();
}

void TW_CALL camera_get_heading (void *value, void *clientData) {
	*static_cast<float *>(value) = (static_cast<Camera *>(clientData)) -> getHeading();
}

void TW_CALL get_value (void *value, void *clientData) {
	*static_cast<float *>(value) = *static_cast<float *>(clientData);
}

void TW_CALL set_value (const void *value, void *clientData) {
	*static_cast<float *>(clientData) = *(const float*)value;
	generation_h0(lx, lz, wind.x, wind.y, A_norm);
}


void TW_CALL savecamera(void *clientData) { 
    if (nameSaved == "") return;

	std::ofstream out(path + nameSaved);
	out << c_main.printMe() << "\n" << c_sec.printMe() << "\n";
	out << resolution << " " << lx << " " << lz << " " << " " << A_norm << " " << wind.x << " " << wind.y << "\n";
}

void TW_CALL loadcamera(void *clientData) { 
    if (nameSaved == "") return;

	std::ifstream in(path + nameSaved);
	std::string mains, secs;
	std::getline(in, mains);
	std::getline(in, secs);
	c_main = Camera(mains);
	c_sec = Camera(secs);
	in >> resolution >> lx >> lz >> A_norm >> wind.x >> wind.y;
	rebuildGrid();
	generation_h0(lx, lz, wind.x, wind.y, A_norm);
}

void TW_CALL geometry_on_off(void *clientData) { 
	if (geometry) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);	
	} else {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);	
	}
	geometry = !geometry;
}

void TW_CALL pause_button(void *clientData) { 
	pause = !pause;
}

void TW_CALL CopyStdStringToClient(std::string& destinationClientString, const std::string& sourceLibraryString) {
  destinationClientString = sourceLibraryString;
}

void initTW () {
	int term = resolution;
	quat rotation;
	TwGLUTModifiersFunc  (glutGetModifiers);
	TwBar *bar = TwNewBar("Parameters");
	TwDefine(" Parameters size='400 600' color='170 30 20' alpha=200 valueswidth=220 text=dark position='20 70' ");
	TwAddVarRW(bar, "Resolution", TW_TYPE_INT32, &resolution,
				" min=1 max=5000 step=10");
	TwAddVarCB(bar, "LX", TW_TYPE_FLOAT, set_value, get_value, &lx,
				NULL);
	TwAddVarCB(bar, "LZ", TW_TYPE_FLOAT, set_value, get_value, &lz,
				NULL);
	TwAddVarCB(bar, "A_norm", TW_TYPE_FLOAT, set_value, get_value, &A_norm,
				NULL);
	TwAddVarCB(bar, "Wind_X", TW_TYPE_FLOAT, set_value, get_value, &wind.x,
				NULL);
	TwAddVarCB(bar, "Wind_Y", TW_TYPE_FLOAT, set_value, get_value, &wind.y,
				NULL);
	
	TwAddButton(bar, "PAUSE", pause_button, NULL, NULL);
	TwAddVarRW(bar, "LightDir", TW_TYPE_DIR3F, &sun_direction,
               " label='Light direction' opened=true  ");

	TwStructMember pointMembers[] = { 
        { "X", TW_TYPE_FLOAT, offsetof(Point, X), NULL},
        { "Y", TW_TYPE_FLOAT, offsetof(Point, Y), NULL },
		{ "Z", TW_TYPE_FLOAT, offsetof(Point, Z), NULL}};
	TwType pointType = TwDefineStruct("POINT", pointMembers, 3, sizeof(Point), NULL, NULL);

	TwAddVarCB(bar, "Pos1", pointType, NULL, camera_get_pos, &c_main, 
				"group='MainCamera' Label='Position'");
	TwAddVarCB(bar, "ObjectOrientation1", TW_TYPE_QUAT4F, NULL, camera_get_rotation, &c_main, 
                   " group='MainCamera' Label='ObjectOrienation' opened=true ");
	TwAddVarCB(bar, "Pitch1", TW_TYPE_FLOAT, NULL, camera_get_pitch, &c_main, "group='MainCamera' Label='Pitch'");
	TwAddVarCB(bar, "Heading1", TW_TYPE_FLOAT, NULL, camera_get_heading, &c_main, "group='MainCamera' Label='Heading'");
	TwAddVarCB(bar, "Roll1", TW_TYPE_FLOAT, NULL, camera_get_roll, &c_main, "group='MainCamera' Label='Roll'");
	TwAddVarCB(bar, "Fov1", TW_TYPE_FLOAT, NULL, camera_get_fov, &c_main, "group='MainCamera' Label='Fov'");

	TwAddVarCB(bar, "Position2", pointType, NULL, camera_get_pos, &c_sec, 
				" group='SecondCamera' Label='Position' ");
	TwAddVarCB(bar, "ObjectOrientation2", TW_TYPE_QUAT4F, NULL, camera_get_rotation, &c_sec, 
                   " group='SecondCamera' Label='ObjectOrienation' opened=true ");
	TwAddVarCB(bar, "Pitch2", TW_TYPE_FLOAT, NULL, camera_get_pitch, &c_sec, "group='SecondCamera' Label='Pitch'");
	TwAddVarCB(bar, "Heading2", TW_TYPE_FLOAT, NULL, camera_get_heading, &c_sec, "group='SecondCamera' Label='Heading'");
	TwAddVarCB(bar, "Roll2", TW_TYPE_FLOAT, NULL, camera_get_roll, &c_sec, "group='SecondCamera' Label='Roll'");
	TwAddVarCB(bar, "Fov2", TW_TYPE_FLOAT, NULL, camera_get_fov, &c_sec, "group='SecondCamera' Label='Fov'");
	TwCopyStdStringToClientFunc(CopyStdStringToClient);
	TwAddVarRW(bar, "Choose file name", TW_TYPE_STDSTRING, &nameSaved, NULL);
	TwAddButton(bar, "Save", savecamera, NULL, NULL);
	TwAddButton(bar, "Load", loadcamera, NULL, NULL);
}

int main(int argc, char ** argv)
{
	fft_init();
	cuda_init_h();
	glutInit(&argc, argv);
	InitializeMagick(NULL);

	glutInitDisplayMode(GLUT_ACCUM | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize (1000, 1000);
	glutCreateWindow( "window" );
	
	TwWindowSize(800, 800);
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(key);
	glutMouseFunc(mouse);
	glutMotionFunc(motionMouse);
	glEnable(GL_DEPTH_TEST);
	glutIdleFunc(idle);
	glewInit();
	TwInit(TW_OPENGL, NULL);
	init();
	initTW();
	glutMainLoop();

	return 0;
}