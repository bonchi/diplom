﻿#include "vars_and_constants.h""

float vecLen(vec2 a) {
	float t = sqrt(a.x * a.x + a.y * a.y);
	return t;
}

float phillipsSpectrum (vec2 k) {
	float t = vecLen(wind);
	if (t == 0) {
		return 0;
	}
	float l = t * t / g;
	float kl = vecLen(k);
	float term = dot(normalize(k),normalize(wind));
	float t1 = exp(-1 / (kl * kl * l * l));
	return A_norm * t1 * term * term / (kl * kl * kl * kl);
}

std::complex<float> get_h0(vec2 k) {
	double ret = (double)rand() / ((double)rand() + 0.1);
	float e1 = (float)ret - floor(ret);
	ret = (double)rand() / ((double)rand() + 0.1);
	float e2 = (float)ret - floor(ret);
	float norm1 = cos(2 * PI * e1) * sqrt(- 2 * log(e2));
	float norm2 = sin(2 * PI * e1) * sqrt(- 2 * log(e2));
	std::complex<float> c(norm1, norm2);
	float term =  (float)sqrt( phillipsSpectrum(k) * 0.5);
	return c * term;
}

void generationH0() {
	int t2 = waves_resolution / 2;
	for (int i = - t2; i <= t2 ; ++i) {
		for (int j = - t2; j <= t2; ++j) {
			if ((i == 0) && (j == 0)) {
				continue;
			}
			vec2 k = vec2(2 * PI * i / lx,2 * PI * j / lz);
			h0[(waves_resolution + 1) * (i + waves_resolution / 2) + (j + waves_resolution / 2)] = get_h0(k);			
		}	
	}
}

std::complex<float> get_h(int i, int j, float t) {
	std::complex<float> h_0 = h0[(waves_resolution + 1) * (i + waves_resolution / 2) + (j + waves_resolution / 2)];
	std::complex<float> h_1 = h0[(waves_resolution + 1) * (-i + waves_resolution / 2) + (-j + waves_resolution / 2)];
	vec2 k = vec2(2 * PI * i / lx,2 * PI * j / lz);
	std::complex<float> p (0, sqrt(g * vecLen(k)) * t);
	return h_0 * exp(p) + std::complex<float>(real(h_1), -imag(h_1)) * exp(-p);
}

void generationHeight(float t) {	
	int t2 = waves_resolution / 2;
	for (int i = - t2; i < t2 ; ++i) {
		for (int j = - t2; j < t2; ++j) {
			if ((i == 0) && (j == 0)) {
				continue;
			}
			std::complex<float> term =  get_h(i , j, t);
			h_koff[(waves_resolution + 1) * (i + waves_resolution / 2) + (j + waves_resolution / 2)].x = term.real();
			h_koff[(waves_resolution + 1) * (i + waves_resolution / 2) + (j + waves_resolution / 2)].y = term.imag();
		}	
	}
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

		generationHeight((float) clock() / CLOCKS_PER_SEC);

		glUseProgram(prg);
		glBindBuffer(GL_ARRAY_BUFFER, buf_tex);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buf_index);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);	
		glDrawElements(GL_TRIANGLES, index.size(), GL_UNSIGNED_INT, NULL);

		for (int i = 0; i < 8; ++i) {
			rc[i] = m_proj2 * cube[i];
		}
		for (int i = 0; i < 4; ++i) {
			vec4 term = rc[2 * i] - rc[2 * i + 1];
			float k = -rc[2 * i + 1].z / term.z;
			trap[i] = rc[2 * i + 1] + k * term;
			float term2 = 0;
		}

		glUniformMatrix4fv(glGetUniformLocation(prg, "m_mvp"), 1, false, value_ptr(m2));
		glUniform4fv(glGetUniformLocation(prg, "trap"), 4, value_ptr(trap[0]));
		glUniform1i(glGetUniformLocation(prg, "waves_resolution"), waves_resolution);
		glUniform1f(glGetUniformLocation(prg, "lx"), lx);
		glUniform1f(glGetUniformLocation(prg, "lz"), lz);
		glUniform2fv(glGetUniformLocation(prg, "h_koff"), MAX_WAVES_RESOLUTION * MAX_WAVES_RESOLUTION, value_ptr(h_koff[0])); 
	}
	assert(glGetError() == GL_NO_ERROR);
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
	srand(time(0));
	generationH0();
	glGenBuffers(1, &buf_tex);
	glGenBuffers(1, &buf_index);
	rebuildGrid();
	glBindBuffer(GL_ARRAY_BUFFER, buf_tex);
	glVertexAttribPointer(1, 2, GL_FLOAT, false, 0, NULL);
	glEnableVertexAttribArray(1);	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	GLuint fshader = glCreateShader(GL_FRAGMENT_SHADER);
	GLuint vshader = glCreateShader(GL_VERTEX_SHADER);
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

	glCompileShader(vshader);
	GLint param;

	glCompileShader(fshader);

	prg = glCreateProgram();
	glAttachShader(prg, vshader);
	glAttachShader(prg, fshader);
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
	
	if (!param)
		throw 1;
}

void reshape(int width, int height) {
   TwWindowSize(width, height);
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
	generationH0();
}


void TW_CALL savecamera(void *clientData) { 
    if (nameSaved == "") return;

	std::ofstream out(path + nameSaved);
	out << c_main.printMe() << "\n" << c_sec.printMe() << "\n";
	out << resolution << " " << lx << " " << lz << " " << waves_resolution << " " << A_norm << " " << wind.x << " " << wind.y << "\n";
}

void TW_CALL loadcamera(void *clientData) { 
    if (nameSaved == "") return;

	std::ifstream in(path + nameSaved);
	std::string mains, secs;
	std::getline(in, mains);
	std::getline(in, secs);
	c_main = Camera(mains);
	c_sec = Camera(secs);
	in >> resolution >> lx >> lz >> waves_resolution >> A_norm >> wind.x >> wind.y;
	rebuildGrid();
	generationH0();
}

void TW_CALL CopyStdStringToClient(std::string& destinationClientString, const std::string& sourceLibraryString) {
  destinationClientString = sourceLibraryString;
}

void initTW () {
	int term = resolution;
	quat rotation;
	TwGLUTModifiersFunc  (glutGetModifiers);
	TwBar *bar = TwNewBar("Parameters");
	TwDefine(" Parameters size='350 600' color='170 30 20' alpha=255 valueswidth=220 text=dark position='20 70' ");
	TwAddVarRW(bar, "Resolution", TW_TYPE_INT32, &resolution,
				" min=1 max=255 step=10");
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
	
	glutInit(&argc, argv);
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