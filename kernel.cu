// Includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <cmath>
#define _USE_MATH_DEFINES
#include <math.h>
#include "glm/gtc/matrix_transform.hpp"
#include <vector>
#include "shape.h"
#include "transformations.h"
#include "shader2.h"
#include<time.h>

using namespace glm;
using namespace std;

/*
    Aquí va el kernel :D
*/

__global__ void kernel_particula(int particles_num, vec3* particlesPositions, vec3* particlesNewPositions, 
    vec3* particlesVelocities, vec3* particlesNewVelocities, float radio) {
    int particleIndex = threadIdx.x + blockIdx.x * blockDim.x;
    bool collision = false;
    for (int i = 0; i < particles_num; i++) {
        vec3 posActual = particlesPositions[particleIndex];
        vec3 posI = particlesPositions[i];
        vec3 velActual = particlesVelocities[particleIndex];
        vec3 velI = particlesVelocities[i];
        if (i == particleIndex) {
            continue;
        }
        double distance = sqrt(pow(posActual[0] - posI[0], 2.0) + pow(posActual[1] -  posI[1], 2.0) + pow(posActual[2] - posI[2], 2.0));
        if (distance <= radio * 2.0f) {
            //printf("choque %d con %d, a %f/n", particleIndex, i, distance);
            collision = true;
            vec3 normal = posI - posActual;
            /*
            normal[0] = posI[0] - posActual[0];
            normal[1] = posI[1] - posActual[1];
            normal[2] = posI[2] - posActual[2];
            */
            //normal = normalize(normal);

            float modulo = sqrt(pow(normal[0], 2) + pow(normal[1], 2) + pow(normal[2], 2));
            normal[0] = normal[0] / modulo;
            normal[1] = normal[1] / modulo;
            normal[2] = normal[2] / modulo;

            vec3 relativeVel = velI - velActual;
            /*
            relativeVel[0] = velI[0] - velActual[0];
            relativeVel[1] = velI[1] - velActual[1];
            relativeVel[2] = velI[2] - velActual[2];
            */
            float prod_punto = normal[0] * relativeVel[0] + normal[1] * relativeVel[1] + normal[2] * relativeVel[2];
            //normal *= prod_punto;
            vec3 normalVelocity = vec3(normal[0] * prod_punto, normal[1] * prod_punto, normal[2] * prod_punto);
            vec3 velocityRes = velActual + normalVelocity;//vec3(velActual[0] + normalVelocity[0], velActual[1] + normalVelocity[1], velActual[2] + normalVelocity[2]);
            particlesNewVelocities[particleIndex] = velocityRes;

            break;
        }
        if(!collision){
            particlesNewVelocities[particleIndex] = velActual;
            particlesNewPositions[particleIndex] = posActual;
        }

    }
}

int particles_num = 100;
const float gravity = -20.8f;
const float radio = 0.07f;
const float coeficienteRoce = 0.8f;
    
/*
    Load an OBJ file. Only triangular faces.
*/

bool loadOBJ(
    const char* path,
    vector < float >& out_vertices,
    vector < unsigned int >& out_indexs,
    vec3 color
){
    vector< unsigned int > vertexIndices, uvIndices, normalIndices;
    vector< vec3 > temp_vertices;
    vector< vec2 > temp_uvs;
    vector< vec3 > temp_normals;

    
    FILE * file = fopen(path, "r");
    if( file == NULL ){
        printf("Impossible to open the file !\n");
        return false;
    }
    unsigned int index = 0;
    while (true) {

        char lineHeader[128];
        // read the first word of the line
        int res = fscanf(file, "%s", lineHeader);
        if (res == EOF) break; // EOF = End Of File. Quit the loop.

        // else : parse lineHeader
        if (strcmp(lineHeader, R"(v)") == 0) {
            vec3 vertex;
            fscanf(file, R"(%f %f %f)", &vertex.x, &vertex.y, &vertex.z);
            temp_vertices.push_back(vertex);
        }
        else if (strcmp(lineHeader, "vt") == 0) {
            vec2 uv;
            fscanf(file, R"(%f %f\n)", &uv.x, &uv.y);
            temp_uvs.push_back(uv);
        }
        else if (strcmp(lineHeader, "vn") == 0) {
            vec3 normal;
            fscanf(file, R"(%f %f %f\n)", &normal.x, &normal.y, &normal.z);
            temp_normals.push_back(normal);
        }
        else if (strcmp(lineHeader, "f") == 0) {
            //string vertex1, vertex2, vertex3;
            unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
            int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0], &uvIndex[0], &normalIndex[0], &vertexIndex[1], &uvIndex[1], &normalIndex[1], &vertexIndex[2], &uvIndex[2], &normalIndex[2]);
            if (matches != 9) {
                printf("File can't be read by our simple parser : ( Try exporting with other options\n");
                return false;
            }
            vertexIndices.push_back(vertexIndex[0]);
            vertexIndices.push_back(vertexIndex[1]);
            vertexIndices.push_back(vertexIndex[2]);
            out_indexs.push_back(index);
            out_indexs.push_back(index+1);
            out_indexs.push_back(index+2);
            normalIndices.push_back(normalIndex[0]);
            normalIndices.push_back(normalIndex[1]);
            normalIndices.push_back(normalIndex[2]);
            index += 3;
        }
    }
    // For each vertex of each triangle
    for (unsigned int i = 0; i < vertexIndices.size(); i++) {
        unsigned int vertexIndex = vertexIndices[i];
        unsigned int normalIndex = normalIndices[i];
        vec3 vertex = temp_vertices[vertexIndex - 1];
        vec3 normal = temp_normals[normalIndex - 1];
        out_vertices.push_back(vertex[0]);
        out_vertices.push_back(vertex[1]);
        out_vertices.push_back(vertex[2]);
        out_vertices.push_back(color[0]);
        out_vertices.push_back(color[1]);
        out_vertices.push_back(color[2]);
        out_vertices.push_back(normal[0]);
        out_vertices.push_back(normal[1]);
        out_vertices.push_back(normal[2]);
        
    }
}


vec3 randomVec3(float vmin,float vmax) {
    float randX = vmin + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(vmax-vmin)));
    float randY = vmin + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(vmax-vmin)));
    float randZ = vmin + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(vmax-vmin)));
    //cout << ' ' << randX << ',' << randY << ',' << randZ << endl;
    return vec3(randX, randY, randZ);
}

vec3 calculateNewPos(vec3 pos0,vec3 vel0, float dt, float a) {
    vec3 newVel = vec3(0.0f, 0.0f, 0.0f);
    newVel[0] = vel0[0] * (1 - coeficienteRoce *dt);
    newVel[1] = (vel0[1] + (a * dt)) * (1 - coeficienteRoce * dt);
    newVel[2] = vel0[2] * (1 - coeficienteRoce * dt);
     //sqrt(pow(vel0, vec3(2, 2, 2)) + aceleracion);
    /*
    float newX = pos0[0] + vel0[0] * dt;
    float newY = pos0[1] + vel0[1] * dt + 0.5*a*pow(dt,2.0);
    float newZ = pos0[2] + vel0[2] * dt;
    */
    //newVel *= dt;
    vec3 newPos = vec3(0.0f, 0.0f, 0.0f);
    newPos[0] = pos0[0] + newVel[0]*dt;
    newPos[1] = pos0[1] + newVel[1]*dt;
    newPos[2] = pos0[2] + newVel[2]*dt;
    return newPos;
}

vec3 calculateTranslate(vec3 vel0, float dt, float a) {

    vec3 newVel = vec3(0.0f, 0.0f, 0.0f);
    newVel[0] = (vel0[0] * (1 - coeficienteRoce * dt))*dt;
    newVel[1] = ((vel0[1] + (a * dt)) * (1 - coeficienteRoce * dt))*dt;
    newVel[2] = (vel0[2] * (1 - coeficienteRoce * dt))*dt;
    //vec3 newPos = vec3(newX,newY,newZ);
    return newVel;
}

GLFWwindow* window;

Shape* square;
Shape* sphere1;
Shape* sphere2;

// The transform being used to draw our shape


// These shader objects wrap the functionality of loading and compiling shaders from files.
Shader vertexShader;
Shader fragmentShader;

// GL index for shader program
GLuint shaderProgram;

// Index of the world matrix in the vertex shader.
GLuint worldMatrixUniform;

// Index of the camera matrix in the vertex shader.
GLuint cameraMatrixUniform;

// Here we store the position, of the camera.


// Store the current dimensions of the viewport.
vec2 viewportDimensions = vec2(800, 600);

// Store the current mouse position.
//vec2 mousePosition;

// Window resize callback
void resizeCallback(GLFWwindow* window, int width, int height){
	glViewport(0, 0, width, height);
    viewportDimensions = vec2(width, height);
}

// This will get called when the mouse moves.
/*
void mouseMoveCallback(GLFWwindow *window, GLdouble mouseX, GLdouble mouseY)
{
    mousePosition = vec2(mouseX, mouseY);
}
*/

/*
    Particles Elements
*/
vector<unsigned int> count(particles_num,0);
vector<vector<unsigned int>> particlesID;
vector<vec3> particlesPosition;
vector<vec3> particlesVelocity;
vector<vec3> particlesColor;
vector<Shape*> particlesShapes;

void particleInit(){
    srand(time(NULL));
    //vector< float > verticesOBJ;
    
    //bool res = loadOBJ("esfera.obj", verticesOBJ, uvs, normals, indicesOBJ);
    vector< float > verticesOBJ;
    vector< unsigned int > indicesOBJ;
    bool res = loadOBJ("esfera.obj", verticesOBJ, indicesOBJ, randomVec3(0.0f, 1.0f));
    for (int i=0;i<particles_num;i++){
        
        particlesPosition.push_back(randomVec3(-1.0f,1.0f));
        particlesVelocity.push_back(randomVec3(-0.5f,0.5f));
        cout << i << ' ' << particlesPosition[i][0] << ',' << particlesPosition[i][1] << ',' << particlesPosition[i][2] << endl;
        //particlesTransforms[i].Translate(vec3(0, 0, -5));
        Shape* sphere = new Shape(verticesOBJ, indicesOBJ, randomVec3(0.0f, 1.0f));
        particlesShapes.push_back(sphere);
        //particlesColor.push_back(randomVec3(0.0f,1.0f));        
    }
}
/*
void genParticle() {
    vector< vec3 > verticesOBJ;
    vector< vec3 > colors;
    vector< vec2 > uvs;
    vector< vec3 > normals; // Won't be used at the moment.
    vector< unsigned int > indicesOBJ;
    bool res = loadOBJ("esfera.obj", verticesOBJ, uvs, normals, indicesOBJ);
    particlesPosition.push_back(vec3(0.5f,0.8f,0.5f));
    particlesVelocity.push_back(vec3(0,0,0));
    Transform3D transform;
    particlesTransforms.push_back(transform);
    particlesTransforms[size(particlesTransforms)-1].SetPosition(particlesPosition[size(particlesPosition) - 1]);
    cout << size(particlesTransforms) - 1 << ' ' << particlesTransforms[size(particlesTransforms) - 1].Position()[0] << ',' << particlesTransforms[size(particlesTransforms) - 1].Position()[1] << ',' << particlesTransforms[size(particlesTransforms) - 1].Position()[2] << endl;
    //particlesTransforms[i].Translate(vec3(0, 0, -5));
    Shape* sphere = new Shape(verticesOBJ, indicesOBJ);
    particlesShapes.push_back(sphere);
    particlesColor.push_back(randomVec3(0.0f, 1.0f));
    particles_num++;
}
*/

int main(int argc, char **argv)
{
	// Initializes the GLFW library
	glfwInit();

	// Initialize window
	GLFWwindow* window = glfwCreateWindow(viewportDimensions.x, viewportDimensions.y, "Proyecto GPU", nullptr, nullptr);

	glfwMakeContextCurrent(window);

	//set resize callback
	glfwSetFramebufferSizeCallback(window, resizeCallback);

    //glfwSetCursorPosCallback(window, mouseMoveCallback);

	// Initializes the glew library
	glewInit();


	// Indices for cube (-1, -1, -1) to (1, 1, 1)
    //    [2]------[6]
	// [3]------[7] |
	//	|  |     |  |
	//	|  |     |  |
	//	| [0]----|-[4]
	// [1]------[5]

	// Create square vertex data.
	vector<float> vertices;
    //-X
    //0
    vertices.push_back(-1.0f);vertices.push_back(-1.0f); vertices.push_back(-1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f);vertices.push_back(1.0f);
    vertices.push_back(-1.0f); vertices.push_back(0.0f); vertices.push_back(0.0f);
    //1
    vertices.push_back(-1.0f);vertices.push_back(-1.0f);vertices.push_back(1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(-1.0f); vertices.push_back(0.0f); vertices.push_back(0.0f);
    //2
    vertices.push_back(-1.0f);vertices.push_back(1.0f); vertices.push_back(-1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f);vertices.push_back(1.0f);
    vertices.push_back(-1.0f); vertices.push_back(0.0f); vertices.push_back(0.0f);
    //3
    vertices.push_back(-1.0f);vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(-1.0f); vertices.push_back(0.0f); vertices.push_back(0.0f);
    //+X
    //4
    vertices.push_back(1.0f);vertices.push_back(-1.0f);vertices.push_back(-1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(-1.0f); vertices.push_back(0.0f); vertices.push_back(0.0f);
    //5
    vertices.push_back(1.0f);vertices.push_back(-1.0f);vertices.push_back(1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f);vertices.push_back(1.0f);
    vertices.push_back(1.0f); vertices.push_back(0.0f); vertices.push_back(0.0f);
    //6
    vertices.push_back(1.0f);vertices.push_back(1.0f);vertices.push_back(-1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(1.0f); vertices.push_back(0.0f); vertices.push_back(0.0f);
    //7
    vertices.push_back(1.0f); vertices.push_back(1.0f);vertices.push_back(1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f);vertices.push_back(1.0f);
    vertices.push_back(1.0f); vertices.push_back(0.0f); vertices.push_back(0.0f);

    //-Y
    //0
    vertices.push_back(-1.0f); vertices.push_back(-1.0f); vertices.push_back(-1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(0.0f); vertices.push_back(-1.0f); vertices.push_back(0.0f);
    //1
    vertices.push_back(-1.0f); vertices.push_back(-1.0f); vertices.push_back(1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(0.0f); vertices.push_back(-1.0f); vertices.push_back(0.0f);
    //4
    vertices.push_back(1.0f); vertices.push_back(-1.0f); vertices.push_back(-1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(0.0f); vertices.push_back(-1.0f); vertices.push_back(0.0f);
    //5
    vertices.push_back(1.0f); vertices.push_back(-1.0f); vertices.push_back(1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(0.0f); vertices.push_back(-1.0f); vertices.push_back(0.0f);

    //+Y
    //2
    vertices.push_back(-1.0f); vertices.push_back(1.0f); vertices.push_back(-1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(0.0f); vertices.push_back(1.0f); vertices.push_back(0.0f);
    //3
    vertices.push_back(-1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(0.0f); vertices.push_back(1.0f); vertices.push_back(0.0f);
    //6
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(-1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(0.0f); vertices.push_back(1.0f); vertices.push_back(0.0f);
    //7
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(0.0f); vertices.push_back(1.0f); vertices.push_back(0.0f);

    //-Z
    //0
    vertices.push_back(-1.0f); vertices.push_back(-1.0f); vertices.push_back(-1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(0.0f); vertices.push_back(0.0f); vertices.push_back(-1.0f);
    //2
    vertices.push_back(-1.0f); vertices.push_back(1.0f); vertices.push_back(-1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(0.0f); vertices.push_back(0.0f); vertices.push_back(-1.0f);
    //4
    vertices.push_back(1.0f); vertices.push_back(-1.0f); vertices.push_back(-1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(0.0f); vertices.push_back(0.0f); vertices.push_back(-1.0f);
    //6
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(-1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(0.0f); vertices.push_back(0.0f); vertices.push_back(-1.0f);

    //+Z
    //1
    vertices.push_back(-1.0f); vertices.push_back(-1.0f); vertices.push_back(1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(0.0f); vertices.push_back(0.0f); vertices.push_back(1.0f);
    //3
    vertices.push_back(-1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(0.0f); vertices.push_back(0.0f); vertices.push_back(1.0f);
    //5
    vertices.push_back(1.0f); vertices.push_back(-1.0f); vertices.push_back(1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(0.0f); vertices.push_back(0.0f); vertices.push_back(1.0f);
    //7
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(1.0f); vertices.push_back(1.0f); vertices.push_back(1.0f);
    vertices.push_back(0.0f); vertices.push_back(0.0f); vertices.push_back(1.0f);

    
	vector<unsigned int> indices;

    indices.push_back(0);
    indices.push_back(1);
    indices.push_back(3);
    indices.push_back(2);

    indices.push_back(4);
    indices.push_back(5);
    indices.push_back(7);
    indices.push_back(6);

    indices.push_back(8);
    indices.push_back(9);
    indices.push_back(11);
    indices.push_back(10);

    indices.push_back(12);
    indices.push_back(13);
    indices.push_back(15);
    indices.push_back(14);

    indices.push_back(16);
    indices.push_back(17);
    indices.push_back(19);
    indices.push_back(18);

    /*
    for (int i = 0; i < 2; i++)
    {
		// 'i' will either be 0 or 1

        // left and right sides
        indices.push_back(i * 4 + 0);
        indices.push_back(i * 4 + 1);
        indices.push_back(i * 4 + 3);
        indices.push_back(i * 4 + 2);

        //top and bottom sides
        indices.push_back(i * 2 + 0);
        indices.push_back(i * 2 + 1);
        indices.push_back(i * 2 + 5);
        indices.push_back(i * 2 + 4);

        //front and back sides
        indices.push_back(i + 0);
        indices.push_back(i + 2);
        indices.push_back(i + 6);
        indices.push_back(i + 4);
    */

    square = new Shape(vertices , indices, vec3(1.0,1.0,1.0));


    //square2 = new Shape(vertices2, indices);

    particleInit();

    //glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec3), &vertices[0], GL_STATIC_DRAW);
    
    // In OpenGL, the Z-Axis points out of the screen.
    // Put the cube 5 units away from the camera.


	// Compile the vertex shader.
	vertexShader.InitFromFile("vertex.glsl", GL_VERTEX_SHADER);

	// Load and compile the fragment shader.
	fragmentShader.InitFromFile("fragment.glsl", GL_FRAGMENT_SHADER);

	// Create a shader program.
	shaderProgram = glCreateProgram();
	
	// Attach the vertex and fragment shaders to our program.
	vertexShader.AttachTo(shaderProgram);
	fragmentShader.AttachTo(shaderProgram);

	// Build shader program.
	glLinkProgram(shaderProgram);

	// After the program has been linked, we can ask it where it put our world matrix and camera matrix


    //cout << "Use WASD to move, and the mouse to look around." << endl;
    cout << "Press escape to exit" << endl;

    //transform.SetPosition();
    cudaError_t err = cudaSuccess;
    cudaError_t cudaStatus;
	// Main Loop
    float camera_theta = -3.0f * M_PI / 4.0f;
	while (!glfwWindowShouldClose(window))
	{

        // Exit when escape is pressed
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            break;
        }

        // Calculate delta time.
        float dt = glfwGetTime();
        // Reset the timer.
        glfwSetTime(0);

        // Get the distance from the center of the screen that the mouse has moved
        //glm::vec2 mouseMovement = mousePosition - (viewportDimensions / 2.0f);


        // Clamp the camera from looking up over 90 degrees.
        

        // Move the cursor to the center of the screen
        glfwSetCursorPos(window, viewportDimensions.x/2, viewportDimensions.y/2);


        // Here we get some input, and use it to move the camera
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            //cameraPosition.Translate(cameraPosition.GetForward() * 5.0f * dt);
        }
        //if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            //transformCube.RotateY(-1.0f * dt);
            //transformParticle1.SetPosition(calculatePosition(transformParticle1.Position(), -0.5f * dt));
        //}
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
           // cameraPosition.Translate(cameraPosition.GetForward() * -5.0f * dt);
        }

        /*if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            transformCube.RotateY(1.0f * dt);
        }*/
        
        // rotate square
        //transform.RotateY(1.0f * dt);



        // Cameras use a transformations matrix just like other renderable objects.
        // When we multiply by a world matrix, we are moving an object from local space to world space.
        // When using a camera, we do the exact opposite. We move everything else from world space into camera local space.
        // To do this we make a matrix that does the inverse of what a world matrix does.


        vec3 viewPos = vec3(0, 2.5, 0);
        vec3 up = vec3(0, 0, 1);
        vec3 at = vec3(0, 0, 0);

        mat4 view = lookAt(viewPos, at, up);
        float aspect = viewportDimensions.x / viewportDimensions.y;
        mat4 projection = perspective(45.0, aspect, 0.1, 100.0);
        mat4 model = uniformScale(1.0);

       
        // Perspective projection expands outward from the camera getting wider, and making things that are far away look smaller.
        /*
                    +-----------+
                    |           |
                    | O         |
        +----+      |         o |
        |*  .|      |           |
        +----+      +-----------+
        */
        // First we move everything
        // Compose view and projection into one matrix to send to the gpu




        // Clear the screen.
        glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


		// Set the current shader program.
		glUseProgram(shaderProgram);

        // Send the camera matrix to the shader
        glUniform3f(glGetUniformLocation(shaderProgram, "La"), 0.56f, 0.91f, 0.89f);
        glUniform3f(glGetUniformLocation(shaderProgram, "Ld"), 1.0f, 1.0f, 1.0f);
        glUniform3f(glGetUniformLocation(shaderProgram, "Ls"), 0.12f, 0.58f, 0.31f);

        glUniform3f(glGetUniformLocation(shaderProgram, "Ka"), 1.0f, 1.0f, 1.0f);
        glUniform3f(glGetUniformLocation(shaderProgram, "Kd"), 1.0f, 1.0f, 1.0f);
        glUniform3f(glGetUniformLocation(shaderProgram, "Ks"), 1.0f, 1.0f, 1.0f);

        glUniform3f(glGetUniformLocation(shaderProgram, "lightPosition"), 0.0f, 8.0f, 0.0f);
        glUniform3f(glGetUniformLocation(shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2]);
        glUniform1ui(glGetUniformLocation(shaderProgram, "shininess"), 5);
        glUniform1f(glGetUniformLocation(shaderProgram, "constantAttenuation"), 0.001f);;
        glUniform1f(glGetUniformLocation(shaderProgram, "linearAttenuation"), 0.1f);
        glUniform1f(glGetUniformLocation(shaderProgram, "quadraticAttenuation"), 0.01f);

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_TRUE, &(projection[0][0]));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_TRUE, &(view[0][0]));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_TRUE, &(model[0][0]));

		// Draw using the worldMatrixUniform
        vec3 color = vec3(0.9f, 0.27f, 0.89f);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "transform"), 1, GL_TRUE, &(identity()[0][0]));
        square->Draw(shaderProgram, GL_QUADS, viewPos, projection, view, model);

        vec3 color3 = vec3(0.75f, 0.34f, 0.96f);
        //cout << &(particles[0].sphere) << endl;
        for (int i = 0; i < particles_num; i++) {
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "transform"), 1, GL_TRUE, &(translate(particlesPosition[i][0], particlesPosition[i][1], particlesPosition[i][2])[0][0]));
            (particlesShapes[i])->Draw(shaderProgram, GL_TRIANGLES, viewPos,projection,view, model);
        }
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "transform"), 1, GL_TRUE, &(identity()[0][0]));
        /*
        Aquí se llama el kernel
        */
        
        vec3* particlesPos = &particlesPosition[0];
        vec3* particlesVel = &particlesVelocity[0];

        vec3* particlesNewPos;
        vec3* particlesNewVel;

        particlesNewPos = (vec3*)malloc(sizeof(vec3) * particles_num);
        particlesNewVel = (vec3*)malloc(sizeof(vec3) * particles_num);

        vec3* d_Pos = NULL;
        err = cudaMalloc((void**)&d_Pos, sizeof(vec3)*particles_num);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        vec3* d_NewPos = NULL;
        err = cudaMalloc((void**)&d_NewPos, sizeof(vec3) * particles_num);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        vec3* d_Vel = NULL;
        err = cudaMalloc((void**)&d_Vel, sizeof(vec3)*particles_num);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        vec3* d_NewVel = NULL;
        err = cudaMalloc((void**)&d_NewVel, sizeof(vec3)*particles_num);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }


        err = cudaMemcpy(d_Pos, particlesPos, sizeof(vec3) * particles_num, cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }


        err = cudaMemcpy(d_Vel, particlesVel, sizeof(vec3)*particles_num, cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        
        kernel_particula <<<1, particles_num>>>(particles_num, d_Pos, d_NewPos, d_Vel, d_NewVel, radio);
       
        cudaStatus = cudaDeviceSynchronize();

        
        err = cudaMemcpy(particlesNewPos, d_NewPos, sizeof(vec3)* particles_num, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
            //exit(EXIT_FAILURE);
        }
        err = cudaMemcpy(particlesNewVel, d_NewVel, sizeof(vec3) * particles_num, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
            //exit(EXIT_FAILURE);
        }
        //vector<vec3> newVectorPor(begin(particlesNewPos), end(particlesNewPos));
        //vector<vec3> newVectorVel(begin(particlesNewVel), end(particlesNewVel));
        
        for (int i=0; i<particles_num;i++) {
            particlesPosition[i] = particlesNewPos[i];
            particlesVelocity[i] = particlesNewVel[i];
        }
        
        for (int i = 0; i < particles_num; i++) {
            
            //particlesTransforms[i].SetPosition(newPos);
            //particlesPosition[i] = particlesTransforms[i].Position();
            vec3 pos = particlesPosition[i];

            //particlesTransforms[i].Translate(vec3(0, 0, -5));
            if (abs(pos[0]) + radio > 2.0f) {
                particlesVelocity[i][0] = -particlesVelocity[i][0];
                pos[0] = (pos[0] / abs(pos[0]) * (1 - radio))*2;
            }
            if (abs(pos[1]) + radio > 2.0f) {
                particlesVelocity[i][1] = -particlesVelocity[i][1];
                pos[1] = (pos[1] / abs(pos[1]) * (1 - radio))*2; 
                if (pos[1] > 0) {
                    //cout << pos[1] << endl;
                }
            }
            if (abs(pos[2]) + radio > 2.0f) {
                particlesVelocity[i][2] = -particlesVelocity[i][2];
                pos[2] = (pos[2] / abs(pos[2]) * (1 - radio))*2;
                

            }
            vec3 newPos = calculateNewPos(pos, particlesVelocity[i], dt, gravity);
            particlesPosition[i] = newPos;
            //cout << i << ' ' << particlesPosition[i][0] << ',' << particlesPosition[i][1] << ',' << particlesPosition[i][2] << endl;

        }
        err = cudaFree(d_Pos);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
            //exit(EXIT_FAILURE);
        }
        err = cudaFree(d_Vel);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
            //exit(EXIT_FAILURE);
        }
        err = cudaFree(d_NewPos);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
            //exit(EXIT_FAILURE);
        }
        err = cudaFree(d_NewVel);

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
            //exit(EXIT_FAILURE);
        }
        err = cudaDeviceReset();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        free(particlesNewPos);
        free(particlesNewVel);
        

	    // Draw all indices in the index buffer
	    //glDrawElements(GL_QUADS, indices.size(), GL_UNSIGNED_INT, (void*)0);

	    // Disable vertex attribute and unbind index buffer.
	    //glDisableVertexAttribArray(0);
	    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);



		// Stop using the shader program.
		glUseProgram(0);

		// Swap the backbuffer to the front.
		glfwSwapBuffers(window);

		// Poll input and window events.
		glfwPollEvents();

	}

	// Free memory from shader program and individual shaders
	glDeleteProgram(shaderProgram);


	// Free memory from shape object
	delete square;

	// Free GLFW memory.
	glfwTerminate();
    

	// End of Program.
	return 0;
}